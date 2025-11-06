"""
# Process all data (uses max transaction_date per user)
python scripts/02_silver_latest_transactions.py

# Process up to specific snapshot date
python scripts/02_silver_latest_transactions.py 2017-03-01

"""

import argparse
import logging
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.window import Window

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

input_path = '/app/datamart/silver/transactions'
output_path = '/app/datamart/silver/latest_transactions'

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process silver latest transactions with snapshot date partitioning'
    )
    parser.add_argument(
        'snapshot_date',
        type=str,
        nargs='?',
        default=None,
        help='Snapshot date in YYYY-MM-DD format (optional, if not provided requires manual specification)'
    )
    
    return parser.parse_args()


def validate_date(date_string):
    """Validate date format."""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def process_latest_transactions(spark, input_path, output_path, snapshot_date=None):
    """
    Process latest transactions for a given snapshot date or all data.
    
    Args:
        spark: SparkSession
        input_path: Path to silver transactions
        output_path: Path to output latest transactions
        snapshot_date: Snapshot date in YYYY-MM-DD format (None to process all data)
    """
    if snapshot_date:
        logger.info(f"Processing snapshot date: {snapshot_date}")
    else:
        logger.info("Processing all data (no snapshot date filter)")
    
    # Load all Silver parquet files
    df_silver = spark.read.parquet(input_path)
    
    # Filter transactions table based on transaction_date <= snapshot date (if provided)
    if snapshot_date:
        df_silver_filtered = (
            df_silver
            .filter(F.to_date(F.col("transaction_date")) <= F.to_date(F.lit(snapshot_date)))
        )
        
        # Add snapshot_date column for partitioning
        df_silver_filtered = df_silver_filtered.withColumn(
            "snapshot_date",
            F.lit(snapshot_date)
        )
    else:
        # Process all data - use max transaction_date per user as snapshot_date
        window_max_date = Window.partitionBy("msno")
        df_silver_filtered = df_silver.withColumn(
            "snapshot_date",
            F.max(F.col("transaction_date")).over(window_max_date)
        )
    
    # Create a new column called 'total_plan_days'
    df_silver_filtered = df_silver_filtered.withColumn(
        "total_plan_days",
        F.datediff(F.col("membership_expire_date"), F.col("transaction_date"))
    )
    
    # Partition by msno (member ID) and order by transaction_date
    window_spec = Window.partitionBy("msno").orderBy(F.col("transaction_date").desc())
    
    # Get max transaction_date within each user
    df_silver_filtered = df_silver_filtered.withColumn(
        "max_transaction_date",
        F.max("transaction_date").over(window_spec)
    )
    
    # Filter rows that match the max transaction date (tied rows)
    df_tied_rows = df_silver_filtered.filter(
        F.col("transaction_date") == F.col("max_transaction_date")
    )
    
    # Define a window for ranking total_plan_days within the tied group
    tied_row_window_rank = Window.partitionBy("msno").orderBy(F.col("total_plan_days").desc())
    
    # Create Total_plan_days column (note: different from total_plan_days)
    df_tied_rows = df_tied_rows.withColumn("Total_plan_days", F.col("total_plan_days"))
    
    # Apply custom ranking logic to select final transaction per user
    df_latest_transaction_final = df_tied_rows.withColumn(
        "custom_rank",
        F.row_number().over(
            Window.partitionBy("msno").orderBy(
                # Priority 1: Check for cancellation at 2nd max plan days
                F.when(
                    (F.col("is_cancel") == 1) & (F.rank().over(tied_row_window_rank) == 2), 
                    1
                ).otherwise(0).desc(),
                # Priority 2: Max plan days (desc)
                F.col("Total_plan_days").desc(),
                # Priority 3: Tie-breaker (transaction_date desc)
                F.col("transaction_date").desc()
            )
        )
    ).filter(F.col("custom_rank") == 1).drop("custom_rank")
    
    # Add year and month columns for partitioning
    df_latest_transaction_final = df_latest_transaction_final.withColumn(
        "year",
        F.year(F.col("snapshot_date"))
    ).withColumn(
        "month",
        F.lpad(F.month(F.col("snapshot_date")).cast("string"), 2, "0")
        #F.month(F.col("snapshot_date"))
    )
    
    # Write output partitioned by year and month
    (
        df_latest_transaction_final
        .write
        .mode("overwrite")
        .partitionBy("year", "month")
        .parquet(output_path)
    )
    
    logger.info(f"Successfully written to: {output_path}")
    
    return df_latest_transaction_final


def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate snapshot date if provided
    if args.snapshot_date and not validate_date(args.snapshot_date):
        logger.error(f"Invalid date format: {args.snapshot_date}. Expected YYYY-MM-DD")
        raise ValueError(f"Invalid date format: {args.snapshot_date}")
    
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("silver_latest_transactions") \
        .config("spark.driver.memory", "4g") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    # Process latest transactions
    process_latest_transactions(
        spark=spark,
        input_path=input_path,
        output_path=output_path,
        snapshot_date=args.snapshot_date
    )
    
    spark.stop()
    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()