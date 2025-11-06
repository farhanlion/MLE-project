#!/usr/bin/env python3
# coding: utf-8
"""
Gold Label Store - Production Script
=====================================
Creates churn labels for the gold layer with support for full/incremental processing.

Usage:
    # Full reprocessing
    python gold_label_prod.py --mode full
    
    # Incremental (only new dates not in gold layer)
    python gold_label_prod.py --mode incremental
    
    # Specific date range
    python gold_label_prod.py --mode range --start-date 2017-01-01 --end-date 2017-03-31
    
    # Single date
    python gold_label_prod.py --mode range --start-date 2017-03-01 --end-date 2017-03-01
"""

import sys
import argparse
import logging
from datetime import datetime
from typing import Optional

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import DataFrame

# ===============================
# Configuration
# ===============================
SILVER_TRANSACTIONS = "/app/datamart/silver/transactions"
SILVER_MAX_EXPIRY = "/app/datamart/silver/max_expiry_transactions"
GOLD_OUTPUT = "/app/datamart/gold/label_store"

SPARK_CONFIGS = {
    "spark.driver.memory": "4g",
    "spark.executor.memory": "4g",
    "spark.sql.shuffle.partitions": "200",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
}

# ===============================
# Logging Setup
# ===============================
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ===============================
# Spark Session
# ===============================
def get_spark_session(app_name: str = "gold_label_store") -> pyspark.sql.SparkSession:
    """Initialize and return Spark session with optimized configs."""
    builder = pyspark.sql.SparkSession.builder.appName(app_name).master("local[*]")
    
    for key, value in SPARK_CONFIGS.items():
        builder = builder.config(key, value)
    
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ===============================
# Data Loading
# ===============================
def load_silver_tables(spark: pyspark.sql.SparkSession) -> dict:
    """Load required silver layer tables."""
    print("📥 Loading Silver layer tables...")
    
    tables = {
        'transactions': spark.read.option("header", True).option("inferSchema", True).parquet(SILVER_TRANSACTIONS),
        'txn_snapshots': spark.read.option("header", True).option("inferSchema", True).parquet(SILVER_MAX_EXPIRY)
    }
    
    print("✅ Silver tables loaded\n")
    return tables


def get_existing_dates(spark: pyspark.sql.SparkSession) -> set:
    """Get set of snapshot_dates already in gold layer."""
    try:
        existing = spark.read.parquet(GOLD_OUTPUT)
        dates = existing.select("snapshot_date").distinct().collect()
        return {row.snapshot_date for row in dates}
    except Exception:
        return set()


def filter_snapshot_dates(
    txn_snapshots: DataFrame,
    mode: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    existing_dates: Optional[set] = None
) -> DataFrame:
    """Filter snapshot dates based on processing mode."""
    
    if mode == "full":
        print("🔄 Mode: FULL reprocessing - all dates")
        return txn_snapshots
    
    elif mode == "incremental":
        if existing_dates:
            print(f"🔄 Mode: INCREMENTAL - excluding {len(existing_dates)} existing dates")
            existing_list = list(existing_dates)
            return txn_snapshots.filter(~F.col("snapshot_date").isin(existing_list))
        else:
            print("🔄 Mode: INCREMENTAL - no existing data, processing all dates")
            return txn_snapshots
    
    elif mode == "range":
        print(f"🔄 Mode: RANGE - processing {start_date} to {end_date}")
        filtered = txn_snapshots.filter(
            (F.col("snapshot_date") >= F.lit(start_date)) &
            (F.col("snapshot_date") <= F.lit(end_date))
        )
        return filtered
    
    return txn_snapshots


# ===============================
# Label Generation
# ===============================
def create_churn_labels(txn: DataFrame, txn_snapshots: DataFrame) -> DataFrame:
    """Create churn labels based on renewal behavior."""
    
    print("🏷️  Creating churn labels...")
    
    # Base: snapshot_date and msno combinations
    churn_base = txn_snapshots.select("snapshot_date", "msno").distinct()
    
    # Candidate renewals: transactions without cancellation
    renewals = txn.where(F.col("is_cancel") == 0).select("msno", "transaction_date")
    
    # Broadcast small renewals table if appropriate
    from pyspark.sql.functions import broadcast
    
    # Join condition: renewal within 30 days of snapshot_date
    cond = (
        (F.col("r.msno") == F.col("c.msno")) &
        (F.col("r.transaction_date") >= F.col("c.snapshot_date")) &
        (F.col("r.transaction_date") <= F.date_add(F.col("c.snapshot_date"), 30))
    )
    
    joined = churn_base.alias("c").join(
        renewals.alias("r"),
        on=cond,
        how="left"
    )
    
    # Aggregate to determine churn status
    result_churn = (
        joined.groupBy(F.col("c.snapshot_date"), F.col("c.msno"))
        .agg(
            F.max(
                F.when(F.col("r.transaction_date").isNotNull(), F.lit(1))
                .otherwise(F.lit(0))
            ).alias("has_renewal_30d")
        )
        .withColumn(
            "is_churn",
            F.when(F.col("has_renewal_30d") == 1, F.lit(0)).otherwise(F.lit(1))
        )
        .select(
            F.col("snapshot_date"),
            F.col("msno"),
            F.col("is_churn")
        )
    )
    
    print("✅ Churn labels created")
    return result_churn


# ===============================
# Main Processing Pipeline
# ===============================
def process_labels(
    spark: pyspark.sql.SparkSession,
    mode: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> None:
    """Main label generation pipeline."""
    
    print("=" * 60)
    print("🗃️  Gold Label Store - Production Pipeline")
    print("=" * 60)
    print()
    
    # Load silver tables
    tables = load_silver_tables(spark)
    
    # Get existing dates for incremental mode
    existing_dates = get_existing_dates(spark) if mode == "incremental" else None
    
    # Filter snapshot dates based on mode
    txn_snapshots = filter_snapshot_dates(
        tables['txn_snapshots'],
        mode,
        start_date,
        end_date,
        existing_dates
    )
    
    # Count snapshots to process
    snapshot_count = txn_snapshots.select("snapshot_date", "msno").distinct().count()
    print(f"📊 Processing {snapshot_count:,} user-snapshot combinations\n")
    
    if snapshot_count == 0:
        print("⚠️  No data to process. Exiting.")
        return
    
    # Create churn labels
    result_churn = create_churn_labels(tables['transactions'], txn_snapshots)
    
    # Calculate and display statistics
    total_records = result_churn.count()
    churned = result_churn.filter(F.col("is_churn") == 1).count()
    churn_rate = (churned / total_records * 100) if total_records > 0 else 0
    
    print(f"\n📈 Label Statistics:")
    print(f"   Total records: {total_records:,}")
    print(f"   Churned: {churned:,}")
    print(f"   Retained: {total_records - churned:,}")
    print(f"   Churn rate: {churn_rate:.2f}%")
    
    # Write to gold layer partitioned by snapshot_date
    print(f"\n💾 Writing to gold layer...")
    print(f"📁 Output: {GOLD_OUTPUT}")
    
    result_churn.write \
        .mode("overwrite") \
        .partitionBy("snapshot_date") \
        .parquet(GOLD_OUTPUT)
    
    print(f"✅ Successfully wrote {total_records:,} records")
    print(f"📅 Partitioned by snapshot_date\n")
    
    # Show sample
    print("📋 Sample output:")
    result_churn.select("snapshot_date", "msno", "is_churn").show(5, truncate=False)
    
    print("=" * 60)
    print("✨ Pipeline completed successfully!")
    print("=" * 60)


# ===============================
# CLI Interface
# ===============================
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gold Label Store - Production Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full reprocessing
  python gold_label_prod.py --mode full
  
  # Incremental (only new dates)
  python gold_label_prod.py --mode incremental
  
  # Specific date range
  python gold_label_prod.py --mode range --start-date 2017-01-01 --end-date 2017-03-31
  
  # Single date
  python gold_label_prod.py --mode range --start-date 2017-03-01 --end-date 2017-03-01
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['full', 'incremental', 'range'],
        help='Processing mode: full (all data), incremental (new dates only), or range (specific dates)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for range mode (YYYY-MM-DD format)'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for range mode (YYYY-MM-DD format)'
    )
    
    args = parser.parse_args()
    
    # Validate range mode arguments
    if args.mode == 'range':
        if not args.start_date or not args.end_date:
            parser.error("--mode range requires both --start-date and --end-date")
        
        # Validate date format
        try:
            datetime.strptime(args.start_date, '%Y-%m-%d')
            datetime.strptime(args.end_date, '%Y-%m-%d')
        except ValueError:
            parser.error("Dates must be in YYYY-MM-DD format")
        
        if args.start_date > args.end_date:
            parser.error("--start-date must be before or equal to --end-date")
    
    return args


def main():
    """Main entry point."""
    args = parse_arguments()
    
    spark = None
    try:
        spark = get_spark_session()
        process_labels(
            spark,
            mode=args.mode,
            start_date=args.start_date,
            end_date=args.end_date
        )
        return 0
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        if spark:
            spark.stop()


if __name__ == "__main__":
    sys.exit(main())