

import argparse
import os
import glob
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from datetime import datetime

# ============================================================================
# Logging Setup
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Load features from feature store
# ============================================================================
def load_features(spark, snapshot_date_str):
    """
    Load features from feature store for the given snapshot_date.
    """
    feature_location = "/app/datamart/gold/inference_feature_store/"

    snapshot_dir = f"{feature_location}snapshot_date={snapshot_date_str}/"
    snapshot_glob = f"{snapshot_dir}*.parquet"

    logger.info(f"Looking for feature files in: {snapshot_dir}")

    parquet_files = glob.glob(snapshot_glob)

    if parquet_files:
        logger.info(f"✅ Found {len(parquet_files)} parquet files for snapshot_date.")
        df = spark.read.parquet(snapshot_dir)
    else:
        logger.warning(f"No files found in partition. Checking entire feature store...")

        all_parquet_files = (
            glob.glob(f"{feature_location}*.parquet") +
            glob.glob(f"{feature_location}*/*.parquet") +
            glob.glob(f"{feature_location}*/*/*.parquet")
        )

        if not all_parquet_files:
            raise FileNotFoundError(
                f"No parquet files found in feature store: {feature_location}"
            )

        logger.info(f"Found {len(all_parquet_files)} parquet files in entire store.")
        features = spark.read.parquet(feature_location)
        df = features.filter(col("snapshot_date") == snapshot_date_str)

    if df.count() == 0:
        raise ValueError(f"No features found for snapshot_date={snapshot_date_str}")

    logger.info(f"✅ Loaded {df.count()} rows for snapshot_date={snapshot_date_str}")
    return df


# ============================================================================
# Load MLflow Model
# ============================================================================
def load_mlflow_model(model_name):
    """
    Load an MLflow model by registered name or full URI.
    """
    mlflow.set_tracking_uri("http://mlflow:5000")

    if model_name.startswith("models:/"):
        model_uri = model_name
    else:
        model_uri = f"models:/{model_name}/Production"

    logger.info(f"Loading MLflow model from: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("✅ MLflow model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load MLflow model: {e}")
        raise


# ============================================================================
# Save predictions to datamart
# ============================================================================
def save_predictions(spark, df_predictions, model_name, snapshot_date_str):
    """
    Save predictions to parquet under:
    datamart/gold/model_predictions/<model_name>/
    """
    base_dir = f"datamart/gold/model_predictions/{model_name}/"
    os.makedirs(base_dir, exist_ok=True)

    filename = f"{model_name}_predictions_{snapshot_date_str.replace('-', '_')}.parquet"
    filepath = os.path.join(base_dir, filename)

    (
        spark.createDataFrame(df_predictions)
            .write.mode("overwrite")
            .parquet(filepath)
    )

    logger.info(f"✅ Predictions saved: {filepath}")


# ============================================================================
# Main Inference Pipeline
# ============================================================================
def main(snapshot_date_str, model_name):

    logger.info("=== Starting Model Inference Job ===")

    # Spark session
    spark = pyspark.sql.SparkSession.builder \
        .appName("inference") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load features
    features_sdf = load_features(spark, snapshot_date_str)

    logger.info("Feature schema:")
    features_sdf.printSchema()

    # Convert to pandas
    features_pdf = features_sdf.toPandas()
    logger.info(f"Converted Spark → Pandas: shape={features_pdf.shape}")

    # Extract feature columns
    # feature_cols = [c for c in features_pdf.columns if c.startswith("fe_")]
    feature_cols = ['tenure_days_at_snapshot', 'registered_via', 'city_clean', 
                'sum_secs_w30', 'active_days_w30', 'complete_rate_w30', 
                'sum_secs_w7', 'engagement_ratio_7_30', 'days_since_last_play', 
                'trend_secs_w30', 'auto_renew_share', 'last_is_auto_renew']
    X_inference = features_pdf[feature_cols]

    # Load MLflow model
    model = load_mlflow_model(model_name)

    # Predict
    y_proba = model.predict_proba(X_inference)[:, 1]

    # Output dataframe
    output = features_pdf[["msno", "snapshot_date"]].copy()
    output["model_name"] = model_name
    output["model_predictions"] = y_proba

    # Save
    save_predictions(spark, output, model_name, snapshot_date_str)

    spark.stop()
    logger.info("=== Inference Job Completed ===")


# ============================================================================
# Entry Point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference pipeline")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True,
                        help="MLflow registered model name OR models:/name/version")

    args = parser.parse_args()

    # Validate date
    try:
        datetime.strptime(args.snapshotdate, "%Y-%m-%d")
    except Exception:
        raise ValueError("snapshotdate must be in YYYY-MM-DD format")

    main(args.snapshotdate, args.modelname)
