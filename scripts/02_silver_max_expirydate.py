"""
# Process all daily snapshots (2015-02-01 to 2017-03-01)
python scripts/02_silver_max_expirydate.py

# Process single snapshot date
python scripts/02_silver_max_expirydate.py 2017-03-01

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
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

input_path = '/app/datamart/silver/transactions'
output_path = '/app/datamart/silver/max_expiry_transactions'

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process max expiry transactions with snapshot date processing'
    )
    parser.add_argument(
        'snapshot_date',
        type=str,
        nargs='?',
        default=None,
        help='Single snapshot date in YYYY-MM-DD format (optional, if not provided processes full range)'
    )
    return parser.parse_args()


def validate_date(date_string):
    """Validate date format."""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def process_max_expiry_transactions(spark, input_path, output_path, snapshot_date=None):
    """
    Process max expiry transactions for snapshot dates.
    
    Args:
        spark: SparkSession
        input_path: Path to silver transactions
        output_path: Path to output max expiry transactions
        snapshot_date: Single snapshot date in YYYY-MM-DD format (None for full range)
    """
    if snapshot_date:
        logger.info(f"Processing single snapshot date: {snapshot_date}")
    else:
        logger.info("Processing full date range: 2015-02-01 to 2017-03-01")
    
    # Load silver transactions
    txn = (spark.read
          .option("header", True)
          .option("inferSchema", True)
          .parquet(input_path))
    
    # Ensure date types
    txn = (
        txn.withColumn("transaction_date", F.to_date("transaction_date"))
           .withColumn("membership_expire_date", F.to_date("membership_expire_date"))
    )
    
    # Build snapshot dates
    if snapshot_date:
        # Single snapshot date
        snapshots = (
            spark.range(1)
                 .select(F.lit(snapshot_date).cast("date").alias("snapshot_date"))
        )
    else:
        # Full range: daily snapshots from 2015-02-01 to 2017-03-01
        snapshots = (
            spark.range(1)
                 .select(
                     F.explode(
                         F.sequence(
                             F.to_date(F.lit("2015-02-01")),
                             F.to_date(F.lit("2017-03-01")),
                             F.expr("interval 1 day")
                         )
                     ).alias("snapshot_date")
                 )
        )
    
    # Cartesian over distinct users × snapshot dates
    msnos = txn.select("msno").distinct()
    grid = msnos.crossJoin(snapshots)
    
    # For each (msno, snapshot_date), find the latest membership_expire_date on/before the snapshot
    candidates = (
        grid.join(txn, on="msno", how="left")
            .where(
                F.col("membership_expire_date").isNotNull()
                & (F.col("membership_expire_date") <= F.col("snapshot_date"))
            )
    )
    
    w = Window.partitionBy("msno", "snapshot_date").orderBy(F.col("membership_expire_date").desc())
    latest_before_or_on = candidates.withColumn("rn", F.row_number().over(w)).where("rn = 1")
    
    # Keep only rows where membership_expire_date == snapshot_date
    txn_snapshots = (
        latest_before_or_on
            .where(F.col("membership_expire_date") == F.col("snapshot_date"))
            .drop("rn")
    )
    
    # Add year and month columns for partitioning
    txn_snapshots = txn_snapshots.withColumn(
        "year",
        F.year(F.col("snapshot_date"))
    ).withColumn(
        "month",
        F.lpad(F.month(F.col("snapshot_date")).cast("string"), 2, "0")
        #F.month(F.col("snapshot_date"))
    )
    
    # Write output partitioned by year and month
    (
        txn_snapshots
        .write
        .mode("overwrite")
        .partitionBy("year", "month")
        .parquet(output_path)
    )
    
    logger.info(f"Successfully written to: {output_path}")
    
    return txn_snapshots


def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate snapshot date if provided
    if args.snapshot_date and not validate_date(args.snapshot_date):
        logger.error(f"Invalid date format: {args.snapshot_date}. Expected YYYY-MM-DD")
        raise ValueError(f"Invalid date format: {args.snapshot_date}")
    
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("silver_max_expiry_transactions") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    # Process max expiry transactions
    process_max_expiry_transactions(
        spark=spark,
        input_path=input_path,
        output_path=output_path,
        snapshot_date=args.snapshot_date
    )
    
    spark.stop()
    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
