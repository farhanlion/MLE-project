"""
🥉 Bronze Layer Update Script
Process raw CSV files into Bronze layer parquet files.

Usage:
    python scripts/process_bronze.py                    # Process all datasets
    python scripts/process_bronze.py --dataset transactions
    python scripts/process_bronze.py --dataset members
    python scripts/process_bronze.py --dataset user_logs
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from typing import Dict, List
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.bronze_processing import (
    initialize_spark,
    read_csv,
    add_year_month_columns,
    add_source_tracking,
    merge_dataframes,
    write_partitioned_parquet,
    validate_dataframe,
    get_processing_summary_enhanced,
    incremental_merge_with_dedup,
    save_summary_with_history,
    cleanup_spark
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/bronze_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> Dict:
    """Load configuration from YAML file."""
    logger.info(f"⚙️ Loading configuration from {config_path}")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_transactions(spark, config: Dict, args=None) -> Dict:
    """
    Process transaction dataset(s) - scalable to handle multiple files.
    
    Can handle:
    - Single file: transactions.csv
    - Multiple files: transactions.csv, transactions_v2.csv, transactions_v3.csv, etc.
    - Uses incremental merge pattern for robustness
    
    Args:
        spark: Active SparkSession
        config: Configuration dictionary
        args: Command line arguments (optional, for --files filtering)
    
    Returns:
        Processing summary dictionary
    """
    logger.info("=" * 80)
    logger.info("🦊 PROCESSING TRANSACTIONS")
    logger.info("=" * 80)
    
    try:
        # Get transaction file path(s) from config
        transaction_paths = config['paths'].get('transaction_files', [])
        
        # Backward compatibility: if single path strings exist, convert to list
        if isinstance(transaction_paths, str):
            transaction_paths = [transaction_paths]
        
        # Also support old 'transactions_v1' and 'transactions_v2' keys
        if not transaction_paths:
            legacy_paths = []
            if 'transactions_v1' in config['paths']:
                legacy_paths.append(config['paths']['transactions_v1'])
            if 'transactions_v2' in config['paths']:
                legacy_paths.append(config['paths']['transactions_v2'])
            transaction_paths = legacy_paths
        
        if not transaction_paths:
            raise ValueError("No transaction paths found in config")
        
        logger.info(f"📖 Found {len(transaction_paths)} transaction file(s) to process:")
        for path in transaction_paths:
            logger.info(f"   - {path}")
        
        # Filter by specific files if --files argument provided
        if hasattr(args, 'files') and args.files:
            logger.info(f"🔍 Filtering to process only: {args.files}")
            filtered_paths = []
            for path in transaction_paths:
                filename = Path(path).name
                if filename in args.files:
                    filtered_paths.append(path)
            
            if not filtered_paths:
                raise ValueError(f"None of the specified files found: {args.files}")
            
            transaction_paths = filtered_paths
            logger.info(f"✓ Will process {len(transaction_paths)} file(s) after filtering")
        
        # Expected columns
        expected_cols = config['bronze']['expected_columns']['transactions']
        
        # Process each file and merge incrementally
        for idx, file_path in enumerate(transaction_paths, 1):
            logger.info(f"\n📂 Processing file {idx}/{len(transaction_paths)}: {file_path}")
            
            # Read file
            df_new = read_csv(spark, file_path)
            
            # Validate schema
            validate_dataframe(df_new, expected_cols, check_nulls=False)
            
            # Add source tracking
            source_name = Path(file_path).name
            df_new = add_source_tracking(df_new, source_name)
            
            # Add year/month columns for partitioning
            df_new_partitioned = add_year_month_columns(df_new, 'transaction_date')
            
            # Use incremental merge (works for first file too!)
            logger.info(f"🔄 Merging into bronze layer...")
            df_final = incremental_merge_with_dedup(
                spark=spark,
                new_data_df=df_new_partitioned,
                bronze_path=config['paths']['bronze_transactions'],
                partition_columns=['year', 'month'],
                dedup_columns=['msno', 'transaction_date', 'payment_method_id', 'plan_list_price'],
                date_column='transaction_date'
            )
            
            # Write back to Bronze with SAFE staging pattern
            logger.info(f"💾 Writing to Bronze layer (staging pattern)...")
            write_partitioned_parquet(
                df_final,
                config['paths']['bronze_transactions'],
                partition_columns=config['bronze']['partitions']['transactions'],
                mode='overwrite',
                use_staging=config['bronze'].get('use_staging', True)
            )
            
            logger.info(f"✅ File {idx}/{len(transaction_paths)} processed successfully")
        
        # Generate final summary
        df_bronze = spark.read.parquet(config['paths']['bronze_transactions'])
        summary = get_processing_summary_enhanced(df_bronze, 'transactions')
        
        logger.info("=" * 80)
        logger.info("✅ All transaction files processed successfully")
        logger.info("=" * 80)
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Transaction processing failed: {str(e)}")
        raise


def process_members(spark, config: Dict) -> Dict:
    """
    Process member dataset.
    
    Args:
        spark: Active SparkSession
        config: Configuration dictionary
    
    Returns:
        Processing summary dictionary
    """
    logger.info("=" * 80)
    logger.info("👥 PROCESSING MEMBERS")
    logger.info("=" * 80)
    
    try:
        # Read members file (flat structure)
        logger.info("📖 Reading members file...")
        df = read_csv(spark, config['paths']['members'])
        
        # Validate schema
        expected_cols = config['bronze']['expected_columns']['members']
        validate_dataframe(df, expected_cols, check_nulls=False)
        
        # Add source tracking
        df = add_source_tracking(df, 'members_v3.csv')
        
        # Add year/month columns (using registration_init_time)
        logger.info("📅 Adding date columns...")
        df_final = add_year_month_columns(df, 'registration_init_time')
        
        # Log statistics about year distribution
        logger.info("📊 Year distribution:")
        year_stats = df_final.groupBy('year').count().orderBy('year')
        year_stats.show(20, truncate=False)
        
        # Write to Bronze with SAFE staging pattern
        logger.info("💾 Writing to Bronze layer (staging pattern)...")
        write_partitioned_parquet(
            df_final,
            config['paths']['bronze_members'],
            partition_columns=config['bronze']['partitions']['members'],
            mode=config['bronze']['write_mode'],
            use_staging=config['bronze'].get('use_staging', True)
        )
        
        # Generate summary
        summary = get_processing_summary_enhanced(df_final, 'members')
        logger.info("✅ Members processing completed successfully")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Members processing failed: {str(e)}")
        raise


def process_user_logs(spark, config: Dict, args=None) -> Dict:
    """
    Process user logs dataset(s) - scalable to handle multiple files.
    
    Can handle:
    - Single file: user_logs.csv
    - Multiple files: user_logs.csv, user_logs_v2.csv, etc.
    - Uses incremental merge pattern for robustness
    
    Args:
        spark: Active SparkSession
        config: Configuration dictionary
    
    Returns:
        Processing summary dictionary
    """
    logger.info("=" * 80)
    logger.info("🎧 PROCESSING USER LOGS")
    logger.info("=" * 80)
    
    try:
        # Get user_logs file path(s) from config
        user_logs_paths = config['paths'].get('user_logs_files', [])
        
        # Backward compatibility: if single path string, convert to list
        if isinstance(user_logs_paths, str):
            user_logs_paths = [user_logs_paths]
        
        # Also support old 'user_logs' key
        if not user_logs_paths and 'user_logs' in config['paths']:
            user_logs_paths = [config['paths']['user_logs']]
        
        if not user_logs_paths:
            raise ValueError("No user_logs paths found in config")
        
        logger.info(f"📖 Found {len(user_logs_paths)} user_logs file(s) to process:")
        for path in user_logs_paths:
            logger.info(f"   - {path}")
        
        # Filter by specific files if --files argument provided
        if hasattr(args, 'files') and args.files:
            logger.info(f"🔍 Filtering to process only: {args.files}")
            filtered_paths = []
            for path in user_logs_paths:
                filename = Path(path).name
                if filename in args.files:
                    filtered_paths.append(path)
            
            if not filtered_paths:
                raise ValueError(f"None of the specified files found: {args.files}")
            
            user_logs_paths = filtered_paths
            logger.info(f"✓ Will process {len(user_logs_paths)} file(s) after filtering")
        
        # Expected columns
        expected_cols = config['bronze']['expected_columns']['user_logs']
        
        # Process each file and merge incrementally
        for idx, file_path in enumerate(user_logs_paths, 1):
            logger.info(f"\n📂 Processing file {idx}/{len(user_logs_paths)}: {file_path}")
            
            # Read file
            df_new = read_csv(spark, file_path)
            
            # Validate schema
            validate_dataframe(df_new, expected_cols, check_nulls=False)
            
            # Add source tracking
            source_name = Path(file_path).name
            df_new = add_source_tracking(df_new, source_name)
            
            # Add year/month columns for partitioning
            df_new_partitioned = add_year_month_columns(df_new, 'date')
            
            # Use incremental merge (works for first file too!)
            logger.info(f"🔄 Merging into bronze layer...")
            df_final = incremental_merge_with_dedup(
                spark=spark,
                new_data_df=df_new_partitioned,
                bronze_path=config['paths']['bronze_user_logs'],
                partition_columns=['year', 'month'],
                dedup_columns=['msno', 'date'],
                date_column='date'
            )
            
            # Write back to Bronze with SAFE staging pattern
            logger.info(f"💾 Writing to Bronze layer (staging pattern)...")
            write_partitioned_parquet(
                df_final,
                config['paths']['bronze_user_logs'],
                partition_columns=config['bronze']['partitions']['user_logs'],
                mode='overwrite',
                use_staging=config['bronze'].get('use_staging', True)
            )
            
            logger.info(f"✅ File {idx}/{len(user_logs_paths)} processed successfully")
        
        # Generate final summary
        df_bronze = spark.read.parquet(config['paths']['bronze_user_logs'])
        summary = get_processing_summary_enhanced(df_bronze, 'user_logs')
        
        logger.info("=" * 80)
        logger.info("✅ All user_logs files processed successfully")
        logger.info("=" * 80)
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ User logs processing failed: {str(e)}")
        raise


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='🥉 Process raw CSV data into Bronze layer parquet files'
    )
    parser.add_argument(
        '--dataset',
        choices=['all', 'transactions', 'members', 'user_logs'],
        default='all',
        help='Specific dataset to process (default: all)'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    parser.add_argument(
        '--memory',
        default='8g',
        help='Spark driver memory (default: 8g)'
    )
    parser.add_argument(
        '--files',
        nargs='+',
        help='Specific file(s) to process (e.g., --files user_logs_v2.csv)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("🥉 BRONZE LAYER UPDATE STARTED")
    logger.info("=" * 80)
    logger.info(f"📦 Dataset: {args.dataset}")
    logger.info(f"💾 Memory allocation: {args.memory}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize
    spark = None
    summaries = []
    
    try:
        # Initialize Spark
        spark = initialize_spark(
            app_name=f"bronze_{args.dataset}",
            driver_memory=args.memory
        )
        
        # Process datasets based on argument
        if args.dataset in ['all', 'transactions']:
            summary = process_transactions(spark, config, args)
            summaries.append(summary)
        
        if args.dataset in ['all', 'members']:
            summary = process_members(spark, config)
            summaries.append(summary)
        
        if args.dataset in ['all', 'user_logs']:
            summary = process_user_logs(spark, config, args)
            summaries.append(summary)
        
        # Save summary report
        save_summary_with_history(summaries)
        
        logger.info("=" * 80)
        logger.info("✅ ALL PROCESSING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 PROCESSING SUMMARY")
        print("=" * 80)
        for summary in summaries:

            required = ["total_rows"]
            missing = [k for k in required if k not in summary]
            if missing:
                raise KeyError(
                    f"Summary missing {missing}. Present: {list(summary.keys())}. "
                    "Ensure your summary builder returns a 'total_rows' key."
                )
            
            print(f"\n📦 Dataset: {summary['dataset_name']}")
            print(f"  📊 Total rows: {summary['total_rows']:,}")
            print(f"  📋 Total columns: {summary['total_columns']}")
            if 'distinct_users' in summary:
                print(f"  👥 Distinct users: {summary['distinct_users']:,}")
            if 'distinct_months' in summary:
                print(f"  📅 Distinct months: {summary['distinct_months']}")
            print(f"  🕐 Processed at: {summary['processing_timestamp']}")
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"❌ Bronze layer update failed: {str(e)}")
        sys.exit(1)
    
    finally:
        if spark:
            cleanup_spark(spark)


if __name__ == "__main__":
    main()