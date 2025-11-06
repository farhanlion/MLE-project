#!/usr/bin/env python3
# coding: utf-8
"""
Gold Label Store - Production Script
=====================================
Creates churn labels for the gold layer.

Usage:
    # Process all data (default)
    python gold_label_prod.py
    
    # Process single date
    python gold_label_prod.py --date 2017-03-01
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


def filter_snapshot_dates(
    txn_snapshots: DataFrame,
    date: Optional[str] = None
) -> DataFrame:
    """Filter snapshot dates based on date parameter."""
    
    if date:
        print(f"🔄 Processing single date: {date}")
        return txn_snapshots.filter(F.col("snapshot_date") == F.lit(date))
    else:
        print("🔄 Processing all dates")
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
    date: Optional[str] = None
) -> None:
    """Main label generation pipeline."""
    
    print("=" * 60)
    print("🗃️  Gold Label Store - Production Pipeline")
    print("=" * 60)
    print()
    
    # Load silver tables
    tables = load_silver_tables(spark)
    
    # Filter snapshot dates based on date
    txn_snapshots = filter_snapshot_dates(
        tables['txn_snapshots'],
        date
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
  # Process all data
  python gold_label_prod.py
  
  # Process single date
  python gold_label_prod.py --date 2017-03-01
        """
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Single date to process (YYYY-MM-DD format). If not specified, processes all dates.'
    )
    
    args = parser.parse_args()
    
    # Validate date format if provided
    if args.date:
        try:
            datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            parser.error("Date must be in YYYY-MM-DD format")
    
    return args


def main():
    """Main entry point."""
    args = parse_arguments()
    
    spark = None
    try:
        spark = get_spark_session()
        process_labels(
            spark,
            date=args.date
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