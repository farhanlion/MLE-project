"""
# Process all bronze data
python scripts/02_silver_transactions.py

# Process specific date partition
python scripts/02_silver_transactions.py 2017-03-01

"""

import argparse
import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process bronze to silver transactions with date-based partition processing'
    )
    parser.add_argument(
        'date',
        type=str,
        nargs='?',
        default=None,
        help='Single date in YYYY-MM-DD format (optional, if not provided processes all data)'
    )
    parser.add_argument(
        '--input-path',
        type=str,
        default='/app/datamart/bronze/transactions/',
        help='Input path for bronze transactions'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='/app/datamart/silver/transactions/',
        help='Output path for silver transactions'
    )
    
    return parser.parse_args()


def validate_date(date_string):
    """Validate date format."""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def process_transactions(spark, input_path, output_path, process_date=None):
    """
    Process bronze transactions to silver layer.
    
    Args:
        spark: SparkSession
        input_path: Path to bronze transactions
        output_path: Path to output silver transactions
        process_date: Single date in YYYY-MM-DD format (None for full processing)
    """
    if process_date:
        # Extract year and month from the provided date
        dt = datetime.strptime(process_date, '%Y-%m-%d')
        year = dt.year
        month = dt.month
        logger.info(f"Processing single partition: year={year}, month={month}")
        
        # Read only the specific partition
        transactions = (
            spark.read
            .option("header", True)
            .parquet(f"{input_path}/year={year}/month={month}")
        )
    else:
        logger.info("Processing all bronze data (full reprocessing)")
        
        # Read all bronze data
        transactions = (
            spark.read
            .option("header", True)
            .parquet(input_path)
        )
    
    # Transform and clean columns
    transactions = (
        transactions
        .withColumn("transaction_date", F.to_date(F.col("transaction_date").cast("string"), "yyyyMMdd"))
        .withColumn("membership_expire_date", F.to_date(F.col("membership_expire_date").cast("string"), "yyyyMMdd"))
        .withColumn("is_auto_renew", F.col("is_auto_renew").cast("int"))
        .withColumn("is_cancel", F.col("is_cancel").cast("int"))
        .withColumn("payment_method_id", F.col("payment_method_id").cast("string"))
        .withColumn("payment_plan_days", F.col("payment_plan_days").cast("int"))
        .withColumn("plan_list_price", F.col("plan_list_price").cast("int"))
        .withColumn("actual_amount_paid", F.col("actual_amount_paid").cast("int"))
        .dropna(subset=["msno", "transaction_date", "membership_expire_date"])
    )
    
    # Add year and month columns for partitioning based on transaction_date
    transactions = transactions.withColumn(
        "year",
        F.year(F.col("transaction_date"))
    ).withColumn(
        "month",
        F.month(F.col("transaction_date"))
    )
    
    # Write to silver layer
    if process_date:
        # When processing single date, use dynamic overwrite mode to overwrite only specific partition
        spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
        
        transactions.write \
            .mode("overwrite") \
            .partitionBy("year", "month") \
            .parquet(output_path)
        
        logger.info(f"Successfully written partition year={year}, month={month} to: {output_path}")
    else:
        # Full reprocessing - overwrite all
        transactions.write \
            .mode("overwrite") \
            .partitionBy("year", "month") \
            .parquet(output_path)
        
        logger.info(f"Successfully written all data to: {output_path}")
    
    return transactions


def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate date if provided
    if args.date and not validate_date(args.date):
        logger.error(f"Invalid date format: {args.date}. Expected YYYY-MM-DD")
        raise ValueError(f"Invalid date format: {args.date}")
    
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("silver_transactions") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    # Process transactions
    process_transactions(
        spark=spark,
        input_path=args.input_path,
        output_path=args.output_path,
        process_date=args.date
    )
    
    spark.stop()
    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
