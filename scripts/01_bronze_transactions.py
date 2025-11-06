"""
Bronze Layer Processing - Transactions
Load transactions CSV into Bronze layer parquet with partitioning.
"""

import logging
import yaml
from datetime import datetime
from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import substring, lit, col

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_spark(app_name: str = "bronze_transactions", driver_memory: str = "8g") -> SparkSession:
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


def incremental_merge_with_dedup(
    spark: SparkSession,
    new_data_df: DataFrame,
    bronze_path: str,
    partition_columns: list,
    dedup_columns: list,
    date_column: str = 'transaction_date'
) -> DataFrame:
    """
    Merge new data with existing bronze layer and deduplicate.
    """
    logger.info("Starting incremental merge with deduplication")
    
    # Identify affected partitions
    affected_partitions = new_data_df.select(*partition_columns).distinct().collect()
    logger.info(f"Affected partitions: {len(affected_partitions)}")
    
    # Try to read existing data
    try:
        df_existing = spark.read.parquet(bronze_path)
        row_count = df_existing.count()
        logger.info(f"Read {row_count:,} existing rows")
        
        # Check if partition columns exist
        existing_columns = set(df_existing.columns)
        missing_partition_cols = [col for col in partition_columns if col not in existing_columns]
        
        if missing_partition_cols:
            logger.info(f"Adding missing partition columns from {date_column}")
            df_existing = add_year_month_columns(df_existing, date_column)
        
        # Schema alignment
        existing_cols = set(df_existing.columns)
        new_cols = set(new_data_df.columns)
        
        missing_in_existing = new_cols - existing_cols
        missing_in_new = existing_cols - new_cols
        
        if missing_in_existing:
            for col_name in missing_in_existing:
                df_existing = df_existing.withColumn(col_name, lit(None))
        
        if missing_in_new:
            for col_name in missing_in_new:
                new_data_df = new_data_df.withColumn(col_name, lit(None))
        
        # Match column order
        final_column_order = new_data_df.columns
        df_existing = df_existing.select(*final_column_order)
        
        # Build filter for affected partitions
        partition_filter = None
        for partition in affected_partitions:
            condition = None
            for col_name in partition_columns:
                col_condition = col(col_name) == partition[col_name]
                condition = col_condition if condition is None else condition & col_condition
            partition_filter = condition if partition_filter is None else partition_filter | condition
        
        # Split existing data
        df_existing_affected = df_existing.filter(partition_filter)
        df_existing_unaffected = df_existing.filter(~partition_filter)
        
        affected_count = df_existing_affected.count()
        unaffected_count = df_existing_unaffected.count()
        logger.info(f"Existing affected: {affected_count:,}, unaffected: {unaffected_count:,}")
        
    except Exception as e:
        logger.info(f"No existing data found: {str(e)}")
        df_existing_affected = spark.createDataFrame([], new_data_df.schema)
        df_existing_unaffected = spark.createDataFrame([], new_data_df.schema)
    
    # Merge and deduplicate
    new_count = new_data_df.count()
    logger.info(f"New data: {new_count:,} rows")
    
    df_merged = df_existing_affected.union(new_data_df)
    merged_count = df_merged.count()
    logger.info(f"After merge: {merged_count:,} rows")
    
    df_deduped = df_merged.dropDuplicates(subset=dedup_columns)
    deduped_count = df_deduped.count()
    duplicates_removed = merged_count - deduped_count
    logger.info(f"After dedup: {deduped_count:,} rows (removed {duplicates_removed:,} duplicates)")
    
    # Combine with unaffected partitions
    df_final = df_existing_unaffected.union(df_deduped)
    final_count = df_final.count()
    logger.info(f"Final dataset: {final_count:,} rows")
    
    return df_final


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
    logger.info("BRONZE LAYER - TRANSACTIONS")
    logger.info("=" * 80)
    

    
    spark = None
    try:
        # Initialize Spark
        spark = initialize_spark()
        
        # Get paths
        transactions_path = "/app/data/transactions.csv"
        transaction_v2_path = "/app/data/transactions_v2.csv"
        bronze_path = "/app/datamart/bronze/transactions"

        # Read CSV
        v1 = read_csv(spark, transactions_path)
        v2 = read_csv(spark, transaction_v2_path)
        df_new = v1.union(v2)
        
        # Add source tracking
        source_name = Path(transactions_path).name
        df_new = add_source_tracking(df_new, source_name)
        
        # Add partitioning columns
        df_new_partitioned = add_year_month_columns(df_new, 'transaction_date')
        
        # Incremental merge with deduplication
        df_final = incremental_merge_with_dedup(
            spark=spark,
            new_data_df=df_new_partitioned,
            bronze_path=bronze_path,
            partition_columns=['year', 'month'],
            dedup_columns=['msno', 'transaction_date', 'payment_method_id', 'plan_list_price'],
            date_column='transaction_date'
        )
        
        # Write to Bronze
        write_partitioned_parquet(
            df_final,
            bronze_path,
            partition_columns=['year', 'month'],
            mode='overwrite'
        )
        
        logger.info("=" * 80)
        logger.info("TRANSACTIONS PROCESSING COMPLETED")
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
