#!/usr/bin/env python3
# coding: utf-8
"""
inference.py

Loads features for a snapshot_date, loads a registered MLflow model and its preprocessing
artifacts, delegates preprocessing to preprocessing_before_fit, runs predict_proba, and
saves predictions back to parquet.

Supports loading model either by registered model name/URI (--modelname) OR by MLflow Run ID (--runid).
If both are provided, --runid takes precedence.
"""

import argparse
import os
import glob
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import pyspark
import pyspark.sql.functions as F
from pyspark.sql.functions import col
from datetime import datetime
import sys
from pathlib import Path

# Add repository root to PYTHONPATH so preprocessing_before_fit can be imported if needed.
repo_root = Path(__file__).resolve().parents[0]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

import preprocessing_before_fit as preproc  # <-- our shared preprocessing module

# ============================================================================#
# Logging Setup
# ============================================================================#
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================#
# Load features from feature store
# ============================================================================#
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
            raise FileNotFoundError(f"No parquet files found in feature store: {feature_location}")

        logger.info(f"Found {len(all_parquet_files)} parquet files in entire store.")
        features = spark.read.parquet(feature_location)
        df = features.filter(col("snapshot_date") == snapshot_date_str)

    if df.count() == 0:
        raise ValueError(f"No features found for snapshot_date={snapshot_date_str}")

    logger.info(f"✅ Loaded {df.count()} rows for snapshot_date={snapshot_date_str}")
    return df


# ============================================================================#
# Load preprocessing artifacts from MLflow run artifacts
# ============================================================================#
def load_preprocessing_artifacts_from_run(run_id):
    """
    Download preprocessing artifact directory and load scaler, lr columns, categorical columns if present.
    Returns (scaler_or_None, lr_feature_cols_or_None, categorical_cols_or_None)
    """
    client = mlflow.tracking.MlflowClient()
    artifact_path = client.download_artifacts(run_id=run_id, path="preprocessing")
    if not artifact_path:
        logger.warning("No preprocessing artifacts directory found in run artifacts.")
        return None, None, None

    scaler = None
    scaler_path = os.path.join(artifact_path, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logger.info("✅ Loaded scaler from MLflow artifacts.")
    else:
        logger.info("No scaler.pkl found in artifacts (likely a tree model).")

    lr_feature_cols = None
    lr_cols_path = os.path.join(artifact_path, "lr_feature_columns.txt")
    if os.path.exists(lr_cols_path):
        with open(lr_cols_path, "r") as f:
            lr_feature_cols = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"✅ Loaded {len(lr_feature_cols)} LR feature columns from artifacts.")

    categorical_cols = None
    cat_cols_path = os.path.join(artifact_path, "categorical_columns.txt")
    if os.path.exists(cat_cols_path):
        with open(cat_cols_path, "r") as f:
            categorical_cols = [line.strip() for line in f.readlines() if line.strip()]
        logger.info(f"✅ Loaded {len(categorical_cols)} categorical columns from artifacts.")

    return scaler, lr_feature_cols, categorical_cols


def load_preprocessing_artifacts(model_uri):
    """
    Given a model URI like 'models:/ModelName/Production' or 'models:/ModelName/1',
    find the run_id from the model version / latest version in stage, and load artifacts.
    Returns scaler, lr_feature_cols, categorical_cols.
    """
    client = mlflow.tracking.MlflowClient()

    if not model_uri.startswith("models:/"):
        raise ValueError("model_uri must start with 'models:/' (e.g. 'models:/MyModel/Production')")

    parts = model_uri.replace("models:/", "").split("/")
    model_name = parts[0]
    version_or_stage = parts[1] if len(parts) > 1 else "Production"

    # If a stage name was provided, get the latest version in that stage; otherwise get specific version
    if not version_or_stage.isdigit():
        # stage name
        versions = client.get_latest_versions(name=model_name, stages=[version_or_stage])
        if not versions:
            raise ValueError(f"No versions found for model {model_name} in stage {version_or_stage}")
        mv = versions[0]
    else:
        mv = client.get_model_version(name=model_name, version=version_or_stage)

    run_id = mv.run_id
    logger.info(f"Resolved model {model_name}/{version_or_stage} -> run_id={run_id}")

    return load_preprocessing_artifacts_from_run(run_id)


# ============================================================================#
# Load MLflow model (sklearn flavor) OR from specific run id
# ============================================================================#
def load_mlflow_model(model_name_or_uri, tracking_uri=None):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        # default expected in your infra
        mlflow.set_tracking_uri("http://mlflow:5000")

    if model_name_or_uri.startswith("models:/"):
        model_uri = model_name_or_uri
    else:
        model_uri = f"models:/{model_name_or_uri}/Production"

    logger.info(f"Loading MLflow model from: {model_uri}")

    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("✅ MLflow model loaded successfully")
        return model, model_uri
    except Exception as e:
        logger.exception("Failed to load MLflow model.")
        raise


def load_model_from_runid(run_id, tracking_uri=None):
    """
    Load a model directly from a run id (artifact path 'model').
    Returns (model, model_uri)
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri("http://mlflow:5000")

    model_uri = f"runs:/{run_id}/model"
    logger.info(f"Loading MLflow model from run URI: {model_uri}")
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("✅ MLflow model loaded from run")
        return model, model_uri
    except Exception as e:
        logger.exception("Failed to load model from run id.")
        raise


# ============================================================================#
# Save predictions to datamart
# ============================================================================#
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


# ============================================================================#
# Inference pipeline
# ============================================================================#
def main(snapshot_date_str, model_name=None, run_id=None, mlflow_tracking_uri=None):
    logger.info("=== Starting Model Inference Job ===")

    # Start Spark
    spark = pyspark.sql.SparkSession.builder \
        .appName("inference") \
        .master("local[*]") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Load features (Spark DF)
    features_sdf = load_features(spark, snapshot_date_str)
    logger.info("Feature schema:")
    features_sdf.printSchema()

    # Convert to pandas
    features_pdf = features_sdf.toPandas()
    logger.info(f"Converted Spark → Pandas: shape={features_pdf.shape}")

    # Make sure the msno and snapshot_date columns exist for output
    if "msno" not in features_pdf.columns or "snapshot_date" not in features_pdf.columns:
        logger.error("Input features must contain 'msno' and 'snapshot_date' columns.")
        raise KeyError("msno or snapshot_date missing in features")

    # Feature subset (same as training)
    feature_cols = [
        'tenure_days_at_snapshot', 'registered_via', 'city_clean',
        'sum_secs_w30', 'active_days_w30', 'complete_rate_w30',
        'sum_secs_w7', 'engagement_ratio_7_30', 'days_since_last_play',
        'trend_secs_w30', 'auto_renew_share', 'last_is_auto_renew'
    ]
    X_inference = features_pdf[feature_cols].copy()

    # Decide how to load model + preprocessing artifacts:
    # If run_id provided -> load model and artifacts from that run
    # Else -> load model by model_name/URI and resolve run -> artifacts
    if run_id:
        model, model_uri = load_model_from_runid(run_id, tracking_uri=mlflow_tracking_uri)
        scaler, lr_feature_cols, categorical_cols = load_preprocessing_artifacts_from_run(run_id)
    else:
        if model_name is None:
            raise ValueError("Either run_id or model_name must be provided.")
        model, model_uri = load_mlflow_model(model_name, tracking_uri=mlflow_tracking_uri)
        scaler, lr_feature_cols, categorical_cols = load_preprocessing_artifacts(model_uri)

    # Apply preprocessing:
    # - If lr_feature_cols and scaler exist -> apply LR-style OHE alignment + scaling
    # - Otherwise apply tree-style preprocessing (missing flag + fill)
    if lr_feature_cols and scaler is not None:
        logger.info("Detected LR-style artifacts (feature column list + scaler). Applying LR preprocessing...")

        # 1) one-hot encode categorical columns (if known). If not known, attempt to infer (no OHE).
        if categorical_cols:
            X_ohe = pd.get_dummies(X_inference.copy(), columns=categorical_cols, drop_first=True, dtype=int)
        else:
            X_ohe = X_inference.copy()

        # 2) align columns to lr_feature_cols (add missing columns with 0, drop extras)
        for c in lr_feature_cols:
            if c not in X_ohe.columns:
                X_ohe[c] = 0
        # Keep only lr_feature_cols in the original order
        X_aligned = X_ohe[lr_feature_cols].copy()

        # 3) scale numeric columns using scaler: detect numeric columns in aligned frame
        numeric_cols = [c for c in X_aligned.columns if pd.api.types.is_numeric_dtype(X_aligned[c])]
        if len(numeric_cols) > 0:
            try:
                X_aligned[numeric_cols] = scaler.transform(X_aligned[numeric_cols])
            except Exception as e:
                logger.exception("Failed to transform numeric columns with scaler. Proceeding without scaling.")
        X_processed = X_aligned.astype(float)

    else:
        logger.info("No LR artifacts detected — applying minimal tree preprocessing.")
        X_processed = preproc.add_missing_flag_and_fill(X_inference)

    logger.info(f"Processed features shape: {X_processed.shape}")

    # Predict probabilities
    try:
        y_proba = model.predict_proba(X_processed)[:, 1]
    except Exception as e:
        logger.exception("model.predict_proba failed. Make sure model supports predict_proba and input shape matches.")
        raise

    # Build output
    output = features_pdf[["msno", "snapshot_date"]].copy()
    output["model_name"] = run_id if run_id else model_name
    output["model_prediction_proba"] = y_proba

    # Save predictions
    save_predictions(spark, output, (run_id if run_id else model_name).replace("models:/", "").replace("/", "_"), snapshot_date_str)

    spark.stop()
    logger.info("=== Inference Job Completed ===")


# ============================================================================#
# CLI
# ============================================================================#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference pipeline")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--runid", type=str, default="2ddc6524115b4d0ebd2d3161c7081289",
                        help="MLflow Run ID to load model/artifacts directly (takes precedence over --modelname)")
    parser.add_argument("--mlflow_tracking_uri", type=str, default=None,
                        help="Optional MLflow tracking URI (if not provided uses http://mlflow:5000)")
    parser.add_argument("--modelpkl", type=str, required=False,
                    help="Path to local .pkl model file (fallback if MLflow not used)")

    args = parser.parse_args()

    # Validate date
    try:
        datetime.strptime(args.snapshotdate, "%Y-%m-%d")
    except Exception:
        raise ValueError("snapshotdate must be in YYYY-MM-DD format")

    main(args.snapshotdate, model_name=args.modelname, run_id=args.runid, mlflow_tracking_uri=args.mlflow_tracking_uri)
