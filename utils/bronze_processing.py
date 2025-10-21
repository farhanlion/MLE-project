"""
🥉 Bronze Layer Processing Utilities
Reusable functions for ingesting raw CSV data into Bronze layer parquet files.
"""

import logging
from typing import List, Optional, Dict
from datetime import datetime
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import substring, lit, col

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def initialize_spark(
    app_name: str = "bronze_processing",
    driver_memory: str = "8g",
    log_level: str = "ERROR"
) -> SparkSession:
    """
    Initialize Spark session with specified configurations.
    
    Args:
        app_name: Name of the Spark application
        driver_memory: Amount of memory allocated to driver
        log_level: Spark log level (ERROR, WARN, INFO, DEBUG)
    
    Returns:
        Configured SparkSession
    """
    logger.info(f"🚀 Initializing Spark session: {app_name}")
    
    spark = SparkSession.builder \
        .appName(app_name) \
        .config("spark.driver.memory", driver_memory) \
        .master("local[*]") \
        .getOrCreate()
    
    spark.sparkContext.setLogLevel(log_level)
    logger.info("✓ Spark session initialized successfully")
    
    return spark


def read_csv(
    spark: SparkSession,
    file_path: str,
    header: bool = True,
    infer_schema: bool = True
) -> DataFrame:
    """
    Read CSV file into Spark DataFrame with error handling.
    
    Args:
        spark: Active SparkSession
        file_path: Path to CSV file (simplified flat structure: data/file.csv)
        header: Whether CSV has header row
        infer_schema: Whether to infer schema automatically
    
    Returns:
        Spark DataFrame
    
    Raises:
        FileNotFoundError: If file doesn't exist
        Exception: For other read errors
    """
    logger.info(f"📖 Reading CSV from: {file_path}")
    
    try:
        df = spark.read.csv(file_path, header=header, inferSchema=infer_schema)
        row_count = df.count()
        logger.info(f"✓ Successfully read {row_count:,} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"✗ Failed to read CSV from {file_path}: {str(e)}")
        raise


def add_year_month_columns(
    df: DataFrame,
    date_column: str,
    year_col_name: str = "year",
    month_col_name: str = "month"
) -> DataFrame:
    """
    Extract year and month from date string column (YYYYMMDD format).
    
    Args:
        df: Input DataFrame
        date_column: Name of column containing date string
        year_col_name: Name for year column
        month_col_name: Name for month column
    
    Returns:
        DataFrame with added year and month columns
    """
    logger.info(f"📅 Adding year/month columns from {date_column}")
    
    df_with_ym = df.withColumn(year_col_name, substring(date_column, 1, 4)) \
                   .withColumn(month_col_name, substring(date_column, 5, 2))
    
    # Log distinct values for validation
    year_count = df_with_ym.select(year_col_name).distinct().count()
    month_count = df_with_ym.select(month_col_name).distinct().count()
    logger.info(f"✓ Found {year_count} distinct years and {month_count} distinct months")
    
    return df_with_ym


def add_source_tracking(
    df: DataFrame,
    source_file: str,
    processing_timestamp: Optional[str] = None
) -> DataFrame:
    """
    Add metadata columns for data lineage tracking.
    
    Args:
        df: Input DataFrame
        source_file: Name of source file
        processing_timestamp: Timestamp of processing (defaults to current time)
    
    Returns:
        DataFrame with tracking columns
    """
    if processing_timestamp is None:
        processing_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"🏷️  Adding source tracking: {source_file}")
    
    return df.withColumn("source_file", lit(source_file)) \
             .withColumn("bronze_ingestion_timestamp", lit(processing_timestamp))


def deduplicate_dataframe(
    df: DataFrame,
    subset_columns: Optional[List[str]] = None
) -> DataFrame:
    """
    Remove duplicate rows from DataFrame.
    
    Args:
        df: Input DataFrame
        subset_columns: List of columns to consider for deduplication.
                       If None, uses all columns.
    
    Returns:
        Deduplicated DataFrame
    """
    initial_count = df.count()
    logger.info(f"🔍 Deduplicating DataFrame with {initial_count:,} rows")
    
    if subset_columns:
        df_deduped = df.dropDuplicates(subset=subset_columns)
    else:
        df_deduped = df.dropDuplicates()
    
    final_count = df_deduped.count()
    removed_count = initial_count - final_count
    
    logger.info(f"✓ Removed {removed_count:,} duplicate rows. Final count: {final_count:,}")
    
    return df_deduped


def merge_dataframes(
    dataframes: List[Dict],
    dedup_columns: Optional[List[str]] = None
) -> DataFrame:
    """
    Merge multiple DataFrames with source tracking and deduplication.
    
    Args:
        dataframes: List of dicts with 'df' and 'source_name' keys
        dedup_columns: Columns to use for deduplication
    
    Returns:
        Merged and deduplicated DataFrame
    
    Example:
        dfs = [
            {'df': df1, 'source_name': 'transactions.csv'},
            {'df': df2, 'source_name': 'transactions_v2.csv'}
        ]
        merged = merge_dataframes(dfs, dedup_columns=['msno', 'transaction_date'])
    """
    logger.info(f"🔀 Merging {len(dataframes)} DataFrames")
    
    # Add source tracking to each DataFrame
    tagged_dfs = []
    for item in dataframes:
        df = item['df']
        source = item['source_name']
        tagged_df = add_source_tracking(df, source)
        tagged_dfs.append(tagged_df)
    
    # Union all DataFrames
    combined = tagged_dfs[0]
    for df in tagged_dfs[1:]:
        combined = combined.union(df)
    
    logger.info(f"Combined DataFrame has {combined.count():,} rows before deduplication")
    
    # Deduplicate
    if dedup_columns:
        merged = deduplicate_dataframe(combined, dedup_columns)
    else:
        merged = deduplicate_dataframe(combined)
    
    return merged


def write_partitioned_parquet(
    df: DataFrame,
    output_path: str,
    partition_columns: List[str],
    mode: str = "overwrite",
    use_staging: bool = True
) -> None:
    """
    Write DataFrame to partitioned Parquet files with optional staging pattern.
    
    Staging pattern prevents data loss:
    1. Write to temporary staging location
    2. Validate the write succeeded
    3. Atomically replace final location
    
    Args:
        df: DataFrame to write
        output_path: Destination path
        partition_columns: Columns to partition by (e.g., ['year', 'month'])
        mode: Write mode (overwrite, append, error, ignore)
        use_staging: Use staging pattern for safe writes (recommended: True)
    """
    import shutil
    from pathlib import Path
    
    logger.info(f"💾 Writing partitioned parquet to: {output_path}")
    logger.info(f"📂 Partitioning by: {partition_columns}")
    logger.info(f"🔧 Write mode: {mode}")
    logger.info(f"🛡️  Using staging: {use_staging}")
    
    if not use_staging:
        # Direct write (original behavior - NOT SAFE!)
        logger.warning("⚠️  Writing without staging - data loss possible if write fails!")
        try:
            if partition_columns:
                df.write \
                    .mode(mode) \
                    .option("header", "true") \
                    .partitionBy(*partition_columns) \
                    .parquet(output_path)
            else:
                df.write \
                    .mode(mode) \
                    .option("header", "true") \
                    .parquet(output_path)
            logger.info(f"✓ Successfully wrote data to {output_path}")
        except Exception as e:
            logger.error(f"✗ Failed to write parquet: {str(e)}")
            raise
        return
    
    # STAGING PATTERN - Safe atomic writes
    staging_path = f"{output_path}_staging_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup_path = None
    
    try:
        # Step 1: Write to staging location
        logger.info(f"📦 Step 1/4: Writing to staging: {staging_path}")
        
        if partition_columns:
            df.write \
                .mode("overwrite") \
                .option("header", "true") \
                .partitionBy(*partition_columns) \
                .parquet(staging_path)
        else:
            df.write \
                .mode("overwrite") \
                .option("header", "true") \
                .parquet(staging_path)
        
        logger.info("   ✓ Staging write completed")
        
        # Step 2: Validate staging data
        logger.info("📋 Step 2/4: Validating staging data...")
        spark = SparkSession.builder.getOrCreate()
        df_validate = spark.read.parquet(staging_path)
        staging_count = df_validate.count()
        original_count = df.count()
        
        if staging_count != original_count:
            raise ValueError(
                f"Staging validation failed! "
                f"Original: {original_count:,} rows, Staging: {staging_count:,} rows"
            )
        
        logger.info(f"   ✓ Validation passed: {staging_count:,} rows")
        
        # Step 3: Backup existing data (if overwriting)
        if mode == "overwrite" and Path(output_path).exists():
            backup_path = f"{output_path}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"💼 Step 3/4: Creating backup: {backup_path}")
            shutil.move(output_path, backup_path)
            logger.info("   ✓ Backup created")
        else:
            logger.info("📝 Step 3/4: No backup needed (first write or append mode)")
        
        # Step 4: Atomic move staging to final location
        logger.info(f"🚀 Step 4/4: Moving staging to final location...")
        shutil.move(staging_path, output_path)
        logger.info(f"   ✓ Successfully moved to {output_path}")
        
        # Clean up backup after successful move
        if backup_path and Path(backup_path).exists():
            logger.info(f"🗑️  Removing backup: {backup_path}")
            shutil.rmtree(backup_path)
        
        logger.info(f"✅ Successfully wrote data to {output_path}")
        
    except Exception as e:
        logger.error(f"❌ Failed to write parquet: {str(e)}")
        
        # Clean up staging on failure
        if Path(staging_path).exists():
            logger.warning(f"🗑️  Cleaning up failed staging: {staging_path}")
            shutil.rmtree(staging_path)
        
        # Restore from backup if we created one
        if backup_path and Path(backup_path).exists():
            logger.warning(f"⚠️  Restoring from backup: {backup_path}")
            shutil.move(backup_path, output_path)
            logger.info("   ✓ Backup restored successfully")
        
        raise


def validate_dataframe(
    df: DataFrame,
    expected_columns: List[str],
    check_nulls: bool = True
) -> bool:
    """
    Validate DataFrame has expected structure and quality.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of columns that must exist
        check_nulls: Whether to check for null values
    
    Returns:
        True if validation passes
    
    Raises:
        ValueError: If validation fails
    """
    logger.info("🔍 Validating DataFrame...")
    
    # Check columns exist
    actual_columns = set(df.columns)
    expected_set = set(expected_columns)
    missing_columns = expected_set - actual_columns
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    logger.info("✓ Column validation passed")
    
    # Check for nulls if requested
    if check_nulls:
        for col_name in expected_columns:
            null_count = df.filter(col(col_name).isNull()).count()
            if null_count > 0:
                logger.warning(f"⚠️  Column '{col_name}' has {null_count:,} null values")
    
    # Basic stats
    total_rows = df.count()
    logger.info(f"✓ DataFrame validation complete. Total rows: {total_rows:,}")
    
    return True


def get_processing_summary(df: DataFrame, dataset_name: str) -> Dict:
    """
    Generate summary statistics for processed dataset.
    
    Args:
        df: Processed DataFrame
        dataset_name: Name of the dataset
    
    Returns:
        Dictionary with summary statistics
    """
    logger.info(f"📊 Generating processing summary for {dataset_name}")
    
    summary = {
        'dataset_name': dataset_name,
        'total_rows': df.count(),
        'total_columns': len(df.columns),
        'columns': df.columns,
        'processing_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Add distinct counts for key columns if they exist
    if 'msno' in df.columns:
        summary['distinct_users'] = df.select('msno').distinct().count()
    
    if 'year' in df.columns and 'month' in df.columns:
        summary['distinct_months'] = df.select('year', 'month').distinct().count()
    
    logger.info(f"✓ Summary: {summary}")
    return summary


def incremental_merge_with_dedup(
    spark: SparkSession,
    new_data_df: DataFrame,
    bronze_path: str,
    partition_columns: List[str],
    dedup_columns: List[str],
    date_column: str = 'date'
) -> DataFrame:
    """
    Robust incremental ingestion with partition-level merge and deduplication.
    
    Handles:
    - Schema evolution (aligns schemas before union)
    - Corrupted/incomplete bronze layers
    - First-time ingestion
    
    Process:
    1. Identify affected partitions from new data
    2. Read existing data (auto-discovers partition columns from folder structure)
    3. If partition columns missing, derive from date column
    4. Align schemas (handle schema evolution)
    5. Merge existing + new data for affected partitions
    6. Deduplicate based on business keys
    7. Combine with unaffected partitions
    8. Return final merged DataFrame
    
    Args:
        spark: Active SparkSession
        new_data_df: DataFrame with new data (must have partition columns)
        bronze_path: Path to bronze layer parquet
        partition_columns: List of partition columns (e.g., ['year', 'month'])
        dedup_columns: Columns to use for deduplication (e.g., ['msno', 'date'])
        date_column: Column name containing date in YYYYMMDD format (default: 'date')
    
    Returns:
        Final merged and deduplicated DataFrame
    """
    logger.info("🔄 Starting incremental merge with deduplication")
    
    # 1. Identify affected partitions from new data
    affected_partitions = new_data_df.select(*partition_columns).distinct().collect()
    logger.info(f"🔍 Affected partitions: {len(affected_partitions)}")
    for partition in affected_partitions:
        partition_str = ", ".join([f"{col}={partition[col]}" for col in partition_columns])
        logger.info(f"   - {partition_str}")
    
    # 2. Try to read existing data with robust error handling
    try:
        logger.info(f"📖 Reading existing data from {bronze_path}")
        
        # Try to read with validation
        try:
            df_existing = spark.read.parquet(bronze_path)
            
            # Validate the read actually works (trigger action)
            row_count = df_existing.count()
            logger.info(f"✓ Successfully read {row_count:,} existing rows")
            
        except Exception as read_error:
            logger.error(f"❌ Failed to read existing bronze layer: {str(read_error)}")
            logger.warning("⚠️  Bronze layer appears corrupted or missing. Treating as first ingestion.")
            raise FileNotFoundError("Corrupted/missing bronze layer")
        
        # Check if partition columns are present
        existing_columns = set(df_existing.columns)
        missing_partition_cols = [col for col in partition_columns if col not in existing_columns]
        
        if missing_partition_cols:
            logger.warning(f"⚠️  Partition columns not in dataframe: {missing_partition_cols}")
            logger.info(f"   Deriving partition columns from '{date_column}' column...")
            df_existing = add_year_month_columns(
                df_existing, 
                date_column,
                year_col_name='year',
                month_col_name='month'
            )
            logger.info(f"✓ Added partition columns to existing data")
        
        # ============================================
        # SCHEMA ALIGNMENT: Handle schema evolution
        # ============================================
        logger.info("🔧 Aligning schemas...")
        
        existing_cols = set(df_existing.columns)
        new_cols = set(new_data_df.columns)
        
        missing_in_existing = new_cols - existing_cols
        missing_in_new = existing_cols - new_cols
        
        if missing_in_existing:
            logger.info(f"   Adding missing columns to existing data: {missing_in_existing}")
            for col_name in missing_in_existing:
                df_existing = df_existing.withColumn(col_name, lit(None))
        
        if missing_in_new:
            logger.info(f"   Adding missing columns to new data: {missing_in_new}")
            for col_name in missing_in_new:
                new_data_df = new_data_df.withColumn(col_name, lit(None))
        
        # Ensure column order matches (union requires matching order)
        final_column_order = new_data_df.columns
        df_existing = df_existing.select(*final_column_order)
        logger.info("✓ Schemas aligned")
        # ============================================
        
        # Build filter for affected partitions
        partition_filter = None
        for partition in affected_partitions:
            condition = None
            for col_name in partition_columns:
                col_condition = col(col_name) == partition[col_name]
                condition = col_condition if condition is None else condition & col_condition
            
            partition_filter = condition if partition_filter is None else partition_filter | condition
        
        # Split existing data: affected vs unaffected partitions
        df_existing_affected = df_existing.filter(partition_filter)
        df_existing_unaffected = df_existing.filter(~partition_filter)
        
        affected_count = df_existing_affected.count()
        unaffected_count = df_existing_unaffected.count()
        
        logger.info(f"✓ Existing data split:")
        logger.info(f"   - Affected partitions: {affected_count:,} rows")
        logger.info(f"   - Unaffected partitions: {unaffected_count:,} rows")
        
    except (FileNotFoundError, Exception) as e:
        logger.warning(f"⚠️  No existing data found or corrupted (treating as first ingestion)")
        logger.warning(f"   Reason: {str(e)}")
        logger.info("   Will perform fresh write with new data only")
        
        # Create empty DataFrames for first ingestion
        df_existing_affected = spark.createDataFrame([], new_data_df.schema)
        df_existing_unaffected = spark.createDataFrame([], new_data_df.schema)
    
    # 3. Merge: existing (affected partition only) + new data
    new_data_count = new_data_df.count()
    logger.info(f"🔥 New data: {new_data_count:,} rows")
    
    df_merged = df_existing_affected.union(new_data_df)
    merged_count = df_merged.count()
    logger.info(f"🔀 After merge: {merged_count:,} rows")
    
    # 4. Deduplicate based on business keys
    logger.info(f"🔍 Deduplicating on columns: {dedup_columns}")
    df_deduped = df_merged.dropDuplicates(subset=dedup_columns)
    deduped_count = df_deduped.count()
    duplicates_removed = merged_count - deduped_count
    logger.info(f"✓ After deduplication: {deduped_count:,} rows")
    logger.info(f"   - Removed {duplicates_removed:,} duplicate rows")
    
    # 5. Combine: unaffected partitions + deduped affected partitions
    df_final = df_existing_unaffected.union(df_deduped)
    final_count = df_final.count()
    logger.info(f"✅ Final dataset: {final_count:,} rows")
    
    return df_final


def cleanup_spark(spark: SparkSession) -> None:
    """
    Safely stop Spark session.
    
    Args:
        spark: SparkSession to stop
    """
    logger.info("🛑 Stopping Spark session...")
    spark.stop()
    logger.info("✓ Spark session stopped successfully")