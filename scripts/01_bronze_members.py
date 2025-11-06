"""
Bronze Layer Processing - Members
Load members CSV into Bronze layer parquet with partitioning.
"""

import logging
import yaml
from datetime import datetime
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import substring, lit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_spark(app_name: str = "bronze_members", driver_memory: str = "8g") -> SparkSession:
    """Initialize Spark session."""
    logger.info(f"Initializing Spark session: {app_name}")
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", driver_memory) \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel("INFO")
    logger.info("Spark session initialized")
    return spark


def read_csv(spark: SparkSession, file_path: str) -> DataFrame:
    """Read CSV file into Spark DataFrame."""
    logger.info(f"Reading CSV: {file_path}")
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    row_count = df.count()
    logger.info(f"Read {row_count:,} rows")
    return df


def add_year_month_columns(df: DataFrame, date_column: str) -> DataFrame:
    """Extract year and month from date column (YYYYMMDD format)."""
    logger.info(f"Adding year/month columns from {date_column}")
    return df.withColumn("year", substring(date_column, 1, 4)) \
             .withColumn("month", substring(date_column, 5, 2))


def add_source_tracking(df: DataFrame, source_file: str) -> DataFrame:
    """Add metadata columns for tracking."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df.withColumn("_source_file", lit(source_file)) \
             .withColumn("_ingestion_timestamp", lit(timestamp))


def write_partitioned_parquet(
    df: DataFrame,
    output_path: str,
    partition_columns: list,
    mode: str = "overwrite"
) -> None:
    """Write DataFrame to partitioned Parquet files."""
    logger.info(f"Writing partitioned parquet to: {output_path}")
    logger.info(f"Partitioning by: {partition_columns}")
    
    if partition_columns:
        df.write \
            .mode(mode) \
            .option("header", "true") \
            .partitionBy(*partition_columns) \
            .option("partitionOverwriteMode", "dynamic") \
            .parquet(output_path)
    else:
        df.write \
            .mode(mode) \
            .option("header", "true") \
            .parquet(output_path)
    
    logger.info(f"Successfully wrote data to {output_path}")



def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("BRONZE LAYER - MEMBERS")
    logger.info("=" * 80)
    
    spark = None
    try:
        # Initialize Spark
        spark = initialize_spark()
        
        # Get paths 
        members_path = "/app/data/members_v3.csv"
        bronze_path = "/app/datamart/bronze/members"
        
        # Read CSV
        df = read_csv(spark, members_path)
        
        # Add source tracking
        source_name = Path(members_path).name
        df = add_source_tracking(df, source_name)
        
        # Add partitioning columns (using registration_init_time)
        df_final = add_year_month_columns(df, 'registration_init_time')
        
        # Write to Bronze
        write_partitioned_parquet(
            df_final,
            bronze_path,
            partition_columns=0,
            mode='overwrite'
        )
        
        logger.info("=" * 80)
        logger.info("MEMBERS PROCESSING COMPLETED")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
    
    finally:
        if spark:
            spark.stop()
            logger.info("Spark session stopped")


if __name__ == "__main__":
    main()
