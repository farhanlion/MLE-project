"""
# Process all daily snapshots (2015-02-01 to 2017-03-01)
python scripts/02_silver_members.py

# Process single snapshot date
python scripts/02_silver_members.py 2017-03-01

"""

import argparse
import logging
from datetime import datetime
import pyspark
import pyspark.sql.functions as F
from pyspark.sql import SparkSession

# Configure logging
logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Process silver members with snapshot date processing'
    )
    parser.add_argument(
        'snapshot_date',
        type=str,
        nargs='?',
        default=None,
        help='Single snapshot date in YYYY-MM-DD format (optional, if not provided processes full range)'
    )
    parser.add_argument(
        '--input-path',
        type=str,
        default='/app/datamart/bronze/members',
        help='Input path for bronze members'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default='/app/datamart/silver/members',
        help='Output path for silver members'
    )
    
    return parser.parse_args()


def validate_date(date_string):
    """Validate date format."""
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def process_members(spark, input_path, output_path, snapshot_date=None):
    """
    Process bronze members to silver layer with snapshot dates.
    
    Args:
        spark: SparkSession
        input_path: Path to bronze members
        output_path: Path to output silver members
        snapshot_date: Single snapshot date in YYYY-MM-DD format (None for full range)
    """
    if snapshot_date:
        logger.info(f"Processing single snapshot date: {snapshot_date}")
    else:
        logger.info("Processing full date range: 2015-02-01 to 2017-03-01")
    
    # Load bronze members
    df_members = (spark.read
                  .option("header", True)
                  .option("inferSchema", True)
                  .parquet(input_path))
    
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
    
    # Cross join members with snapshot dates
    df_members_snapshots = df_members.crossJoin(snapshots)
    
    # ========= 1) Field Format =========
    dfm = (
        df_members_snapshots
          .withColumn("msno", F.lower(F.trim(F.col("msno"))))
          .withColumn("city", F.col("city").cast("int"))
          .withColumn("registered_via", F.col("registered_via").cast("int"))
          .withColumn("registration_init_time", F.col("registration_init_time").cast("string"))
          .drop("bd", "gender")
    )
    
    # ========= 2) Date =========
    dfm = dfm.withColumn(
        "registration_date",
        F.to_date(F.col("registration_init_time"), "yyyyMMdd")
    )
    
    # Filter: only keep members where registration_date <= snapshot_date
    dfm = dfm.filter(F.col("registration_date") <= F.col("snapshot_date"))
    
    # ========= 3) City clean =========
    dfm = dfm.withColumn(
        "city_clean",
        F.when(F.col("city") <= 0, None).otherwise(F.col("city"))
    )
    
    # ========= 4) Tenure to snapshot =========
    dfm = dfm.withColumn(
        "tenure_days_at_snapshot",
        F.datediff(F.col("snapshot_date"), F.col("registration_date"))
    )
    
    # ========= 5) Frequency enrich =========
    # 5a) registered_via frequency (per snapshot_date)
    total_cnt_per_snapshot = dfm.groupBy("snapshot_date").agg(F.count("*").alias("total_count"))
    
    via_freq = (
        dfm.groupBy("snapshot_date", "registered_via")
           .agg(F.count("*").alias("via_count"))
           .join(total_cnt_per_snapshot, on="snapshot_date", how="left")
           .withColumn("registered_via_freq", F.col("via_count") / F.col("total_count"))
           .select("snapshot_date", "registered_via", "registered_via_freq")
    )
    
    # 5b) city frequency (per snapshot_date)
    city_freq = (
        dfm.groupBy("snapshot_date", "city_clean")
           .agg(F.count("*").alias("city_count"))
           .join(total_cnt_per_snapshot, on="snapshot_date", how="left")
           .withColumn("city_freq", F.col("city_count") / F.col("total_count"))
           .select("snapshot_date", "city_clean", "city_freq")
    )
    
    # 5c) Join freq
    dfm = (
        dfm.drop("registered_via_freq", "city_freq")
           .join(via_freq, on=["snapshot_date", "registered_via"], how="left")
           .join(city_freq, on=["snapshot_date", "city_clean"], how="left")
           .fillna({"registered_via_freq": 0.0, "city_freq": 0.0})
    )
    
    # ========= 6) SILVER (clean + enrich) =========
    silver_cols = [
        "snapshot_date",
        "msno",
        "city_clean",
        "registered_via",
        "registration_date",
        "tenure_days_at_snapshot",
        "registered_via_freq",
        "city_freq"
    ]
    silver_members = dfm.select(*silver_cols)
    
    # Add year and month columns for partitioning
    silver_members = silver_members.withColumn(
        "year",
        F.year(F.col("snapshot_date"))
    ).withColumn(
        "month",
        #F.month(F.col("snapshot_date"))
        F.lpad(F.month(F.col("snapshot_date")).cast("string"), 2, "0")
    )
    
    # Write output partitioned by year and month
    if snapshot_date:
        # Use dynamic partition overwrite mode
        spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")
    
    (
        silver_members
        .write
        .mode("overwrite")
        .partitionBy("year", "month")
        .option("compression", "snappy")
        .parquet(output_path)
    )
    
    logger.info(f"Successfully written to: {output_path}")
    
    return silver_members


def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate snapshot date if provided
    if args.snapshot_date and not validate_date(args.snapshot_date):
        logger.error(f"Invalid date format: {args.snapshot_date}. Expected YYYY-MM-DD")
        raise ValueError(f"Invalid date format: {args.snapshot_date}")
    
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("silver_members") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("ERROR")
    
    # Process members
    process_members(
        spark=spark,
        input_path=args.input_path,
        output_path=args.output_path,
        snapshot_date=args.snapshot_date
    )
    
    spark.stop()
    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
