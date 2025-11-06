#!/usr/bin/env python3
# coding: utf-8
"""
Gold Feature Store - Production Script
=======================================
Creates gold layer features with support for full/incremental processing.

Usage:
    # Full reprocessing
    python gold_feature_store_prod.py --mode full
    
    # Incremental (only new dates not in gold layer)
    python gold_feature_store_prod.py --mode incremental
    
    # Specific date range
    python gold_feature_store_prod.py --mode range --start-date 2017-01-01 --end-date 2017-03-31
    
    # Single date
    python gold_feature_store_prod.py --mode range --start-date 2017-03-01 --end-date 2017-03-01
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from typing import Optional

import pyspark
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window
from pyspark.sql.types import DateType

# ===============================
# Configuration
# ===============================
SILVER_USER_LOGS = "/app/datamart/silver/user_logs"
SILVER_TRANSACTIONS = "/app/datamart/silver/transactions"
SILVER_LATEST_TRANSACTIONS = "/app/datamart/silver/latest_transactions"
SILVER_MEMBERS = "/app/datamart/silver/members"
SILVER_MAX_EXPIRY = "/app/datamart/silver/max_expiry_transactions"
GOLD_OUTPUT = "/app/datamart/gold/feature_store"

# Spark configuration
SPARK_CONFIGS = {
    "spark.driver.memory": "4g",
    "spark.executor.memory": "4g",
    "spark.sql.shuffle.partitions": "200",
    "spark.sql.adaptive.enabled": "true",
    "spark.sql.adaptive.coalescePartitions.enabled": "true",
}

# Metrics to aggregate from user logs
METRICS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs"]

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
def get_spark_session(app_name: str = "gold_feature_store") -> pyspark.sql.SparkSession:
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
    """Load all required silver layer tables."""
    print("📥 Loading Silver layer tables...")
    
    tables = {
        'user_logs': spark.read.parquet(SILVER_USER_LOGS),
        'transactions': spark.read.parquet(SILVER_TRANSACTIONS),
        'latest_transactions': spark.read.parquet(SILVER_LATEST_TRANSACTIONS),
        'members': spark.read.parquet(SILVER_MEMBERS),
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
            # Convert set to list for filtering
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
# Feature Engineering - User Logs
# ===============================
def create_userlog_features(df_userlogs: DataFrame, snapshot_users: DataFrame) -> DataFrame:
    """Create user log features with optimized joins and aggregations."""
    
    print("🎵 Creating user log features...")
    
    # Ensure date types
    df_userlogs = df_userlogs.withColumn("date", F.to_date("date"))
    snapshot_users = snapshot_users.withColumn("snapshot_date", F.to_date("snapshot_date"))
    
    # Cache snapshot users for multiple joins
    snapshot_users_cached = snapshot_users.select("msno", "snapshot_date").cache()
    
    # Broadcast small lookup table for better join performance
    from pyspark.sql.functions import broadcast
    
    u = df_userlogs.alias("u")
    s = snapshot_users_cached.alias("s")
    
    # Define date windows
    start_30 = F.date_sub(F.col("s.snapshot_date"), 30)
    end_30 = F.date_sub(F.col("s.snapshot_date"), 1)
    start_7 = F.date_sub(F.col("s.snapshot_date"), 7)
    end_7 = F.date_sub(F.col("s.snapshot_date"), 1)
    
    # Build 30-day windowed user logs
    userlogs_30d = (
        u.join(
            broadcast(s),
            (F.col("u.msno") == F.col("s.msno")) &
            (F.col("u.date").between(start_30, end_30)),
            "inner"
        )
    )
    
    # 30-day aggregates
    agg_30d = (
        userlogs_30d
        .groupBy(F.col("s.msno").alias("msno"), F.col("s.snapshot_date").alias("snapshot_date"))
        .agg(
            *[F.sum(F.col(f"u.{m}")).alias(f"{m}_w30_sum") for m in METRICS],
            F.countDistinct(F.col("u.date")).alias("active_days_w30")
        )
        .withColumn(
            "complete_rate_w30",
            F.col("num_100_w30_sum") / F.when(F.col("num_unq_w30_sum") > 0, F.col("num_unq_w30_sum")).otherwise(F.lit(1))
        )
        .withColumnRenamed("total_secs_w30_sum", "sum_secs_w30")
    )
    
    # Build 7-day windowed user logs
    userlogs_7d = (
        u.join(
            broadcast(s),
            (F.col("u.msno") == F.col("s.msno")) &
            (F.col("u.date").between(start_7, end_7)),
            "inner"
        )
    )
    
    # 7-day aggregates
    agg_7d = (
        userlogs_7d
        .groupBy(F.col("s.msno").alias("msno"), F.col("s.snapshot_date").alias("snapshot_date"))
        .agg(F.sum(F.col("u.total_secs")).alias("sum_secs_w7"))
    )
    
    # Engagement ratio 7/30
    engagement = (
        agg_30d.select("msno", "snapshot_date", "sum_secs_w30")
        .join(agg_7d, ["msno", "snapshot_date"], "left")
        .withColumn(
            "engagement_ratio_7_30",
            F.col("sum_secs_w7") / F.when(F.col("sum_secs_w30") > 0, F.col("sum_secs_w30")).otherwise(F.lit(1))
        )
        .select("msno", "snapshot_date", "engagement_ratio_7_30")
    )
    
    # Days since last play
    last_play = (
        u.join(
            broadcast(s),
            (F.col("u.msno") == F.col("s.msno")) & (F.col("u.date") <= F.col("s.snapshot_date")),
            "inner"
        )
        .groupBy(F.col("s.msno").alias("msno"), F.col("s.snapshot_date").alias("snapshot_date"))
        .agg(F.max(F.col("u.date")).alias("last_play_date"))
        .withColumn("days_since_last_play", F.datediff(F.col("snapshot_date"), F.col("last_play_date")))
        .select("msno", "snapshot_date", "days_since_last_play")
    )
    
    # Trend in total_secs over 30 days
    daily_secs_30d = (
        userlogs_30d
        .groupBy(
            F.col("s.msno").alias("msno"),
            F.col("s.snapshot_date").alias("snapshot_date"),
            F.col("u.date").alias("date")
        )
        .agg(F.sum(F.col("u.total_secs")).alias("daily_secs"))
    )
    
    w = Window.partitionBy("msno", "snapshot_date").orderBy("date")
    trend_30d = (
        daily_secs_30d
        .withColumn("day_idx", F.row_number().over(w))
        .groupBy("msno", "snapshot_date")
        .agg(
            (F.covar_pop("day_idx", "daily_secs") / 
             F.when(F.var_pop("day_idx") > 0, F.var_pop("day_idx")).otherwise(F.lit(1))
            ).alias("trend_secs_w30")
        )
    )
    
    # Assemble all features
    features = (
        agg_30d
        .join(agg_7d, ["msno", "snapshot_date"], "left")
        .join(engagement, ["msno", "snapshot_date"], "left")
        .join(last_play, ["msno", "snapshot_date"], "left")
        .join(trend_30d, ["msno", "snapshot_date"], "left")
    )
    
    # Fill nulls
    fill_map = {
        "sum_secs_w30": 0.0,
        "active_days_w30": 0,
        "complete_rate_w30": 0.0,
        "sum_secs_w7": 0.0,
        "engagement_ratio_7_30": 0.0,
        "days_since_last_play": 0.0,
        "trend_secs_w30": 0.0
    }
    features = features.na.fill(fill_map)
    
    # Unpersist cached data
    snapshot_users_cached.unpersist()
    
    print("✅ User log features created")
    return features


# ===============================
# Feature Engineering - Transactions
# ===============================
def create_transaction_features(
    df_transactions: DataFrame,
    registered_users: DataFrame
) -> DataFrame:
    """Create transaction-based features."""
    
    print("💳 Creating transaction features...")
    
    # Ensure date types
    df_transactions = df_transactions.withColumn("transaction_date", F.to_date("transaction_date"))
    registered_users = registered_users.withColumn("snapshot_date", F.to_date("snapshot_date"))
    registered_users = registered_users.withColumn("registration_date", F.to_date("registration_date"))
    
    t = df_transactions.alias("t")
    r = registered_users.select("msno", "snapshot_date", "registration_date").alias("r")
    
    # Join transactions up to snapshot date
    tx_asof_snap = (
        t.join(r, F.col("t.msno") == F.col("r.msno"), "inner")
        .where(F.col("t.transaction_date") <= F.col("r.snapshot_date"))
    )
    
    # Latest transaction as of snapshot date
    w_latest = Window.partitionBy("r.msno", "r.snapshot_date").orderBy(F.col("t.transaction_date").desc())
    latest_tx = (
        tx_asof_snap
        .withColumn("rn", F.row_number().over(w_latest))
        .where(F.col("rn") == 1)
        .select(
            F.col("r.msno").alias("msno"),
            F.col("r.snapshot_date").alias("snapshot_date"),
            F.col("t.transaction_date").alias("latest_transaction_date"),
            F.col("t.is_auto_renew").alias("last_is_auto_renew"),
            F.col("t.plan_list_price").alias("last_plan_list_price")
        )
    )
    
    # Tenure days
    tenure_asof = latest_tx.join(
        r.select(F.col("msno"), F.col("snapshot_date"), F.col("registration_date")),
        ["msno", "snapshot_date"],
        "left"
    ).withColumn(
        "tenure_days",
        F.datediff(F.col("latest_transaction_date"), F.col("registration_date"))
    )
    
    # Auto-renew statistics
    auto_renew_stats = (
        tx_asof_snap
        .groupBy(F.col("r.msno").alias("msno"), F.col("r.snapshot_date").alias("snapshot_date"))
        .agg(
            F.sum(F.when(F.col("t.is_auto_renew") == 1, 1).otherwise(0)).alias("auto_renew_count"),
            F.count(F.lit(1)).alias("total_tx_before_expire")
        )
        .withColumn(
            "auto_renew_share",
            F.col("auto_renew_count") / 
            F.when(F.col("total_tx_before_expire") > 0, F.col("total_tx_before_expire")).otherwise(F.lit(1))
        )
        .select("msno", "snapshot_date", "auto_renew_share")
    )
    
    # Merge transaction features
    result = (
        registered_users
        .join(
            tenure_asof.select("msno", "snapshot_date", "tenure_days", "last_is_auto_renew", "last_plan_list_price"),
            ["msno", "snapshot_date"],
            "left"
        )
        .join(auto_renew_stats, ["msno", "snapshot_date"], "left")
        .na.fill({
            "tenure_days": 0,
            "last_is_auto_renew": 0,
            "last_plan_list_price": 0.0,
            "auto_renew_share": 0.0
        })
    )
    
    # Drop intermediate columns
    cols_to_drop = ["num_25_w30_sum", "num_50_w30_sum", "num_75_w30_sum", "num_985_w30_sum", "num_100_w30_sum"]
    result = result.drop(*cols_to_drop)
    
    print("✅ Transaction features created")
    return result


# ===============================
# Main Processing Pipeline
# ===============================
def process_features(
    spark: pyspark.sql.SparkSession,
    mode: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> None:
    """Main feature engineering pipeline."""
    
    print("=" * 60)
    print("🗃️  Gold Feature Store - Production Pipeline")
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
    
    # Create snapshot users
    snapshot_users = txn_snapshots.select("snapshot_date", "msno").distinct()
    snapshot_users = snapshot_users.join(tables['members'], on="msno", how="left")
    
    snapshot_count = snapshot_users.count()
    print(f"📊 Processing {snapshot_count:,} user-snapshot combinations\n")
    
    if snapshot_count == 0:
        print("⚠️  No data to process. Exiting.")
        return
    
    # Create user log features
    userlog_features = create_userlog_features(tables['user_logs'], snapshot_users)
    
    # Join user log features to snapshot users
    registered_users = snapshot_users.join(userlog_features, on=["msno", "snapshot_date"], how="left")
    
    # Create transaction features
    final_features = create_transaction_features(tables['transactions'], registered_users)
    
    # Write to gold layer partitioned by snapshot_date
    print(f"\n💾 Writing to gold layer...")
    print(f"📁 Output: {GOLD_OUTPUT}")
    
    final_features.write \
        .mode("overwrite") \
        .partitionBy("snapshot_date") \
        .parquet(GOLD_OUTPUT)
    
    final_count = final_features.count()
    print(f"✅ Successfully wrote {final_count:,} records")
    print(f"📅 Partitioned by snapshot_date\n")
    
    # Show sample
    print("📋 Sample output:")
    final_features.select(
        "msno", "snapshot_date", "sum_secs_w30", "active_days_w30",
        "engagement_ratio_7_30", "tenure_days", "last_is_auto_renew"
    ).show(5, truncate=False)
    
    print("=" * 60)
    print("✨ Pipeline completed successfully!")
    print("=" * 60)


# ===============================
# CLI Interface
# ===============================
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Gold Feature Store - Production Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full reprocessing
  python gold_feature_store_prod.py --mode full
  
  # Incremental (only new dates)
  python gold_feature_store_prod.py --mode incremental
  
  # Specific date range
  python gold_feature_store_prod.py --mode range --start-date 2017-01-01 --end-date 2017-03-31
  
  # Single date
  python gold_feature_store_prod.py --mode range --start-date 2017-03-01 --end-date 2017-03-01
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
        process_features(
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