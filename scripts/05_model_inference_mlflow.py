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
def load_preprocessing_artifacts(model_uri):
    """
    Load preprocessing artifacts (scaler, feature columns, categorical columns)
    from MLflow model artifacts.
    Returns: scaler, lr_feature_cols, categorical_cols
    """
    # Extract run_id and artifact_path from model_uri
    # model_uri format: "models:/ModelName/Production" or "models:/ModelName/1"
    client = mlflow.tracking.MlflowClient()
    
    if model_uri.startswith("models:/"):
        # Parse registered model
        parts = model_uri.replace("models:/", "").split("/")
        model_name = parts[0]
        version_or_stage = parts[1] if len(parts) > 1 else "Production"
        
        # Get model version details
        if version_or_stage.lower() in ["production", "staging", "archived", "none"]:
            model_versions = client.get_latest_versions(model_name, stages=[version_or_stage])
            if not model_versions:
                raise ValueError(f"No model found in stage: {version_or_stage}")
            model_version = model_versions[0]
        else:
            # Specific version number
            model_version = client.get_model_version(model_name, version_or_stage)
        
        run_id = model_version.run_id
    else:
        raise ValueError(f"Unsupported model_uri format: {model_uri}")
    
    logger.info(f"Loading preprocessing artifacts from run_id: {run_id}")
    
    # Download artifacts
    artifact_path = client.download_artifacts(run_id, "preprocessing")
    
    # Load scaler
    scaler = None
    scaler_path = os.path.join(artifact_path, "scaler.pkl")
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        logger.info("✅ Loaded scaler from MLflow")
    else:
        logger.warning("⚠️  No scaler.pkl found - model might be tree-based")
    
    # Load feature columns
    lr_feature_cols = None
    feature_cols_path = os.path.join(artifact_path, "lr_feature_columns.txt")
    if os.path.exists(feature_cols_path):
        with open(feature_cols_path, 'r') as f:
            lr_feature_cols = [line.strip() for line in f.readlines()]
        logger.info(f"✅ Loaded {len(lr_feature_cols)} feature columns from MLflow")
    
    # Load categorical columns
    categorical_cols = None
    cat_cols_path = os.path.join(artifact_path, "categorical_columns.txt")
    if os.path.exists(cat_cols_path):
        with open(cat_cols_path, 'r') as f:
            categorical_cols = [line.strip() for line in f.readlines()]
        logger.info(f"✅ Loaded {len(categorical_cols)} categorical columns from MLflow")
    
    return scaler, lr_feature_cols, categorical_cols


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
# Apply Preprocessing (same as training)
# ============================================================================
def apply_preprocessing(df, categorical_cols=None, scaler=None, lr_feature_cols=None):
    """
    Apply the same preprocessing steps as training:
    Step 1: Create missing value indicators
    Step 2: Fill missing values
    Step 3: One-hot encode categorical features (if applicable)
    Step 4: Scale features (if scaler provided)
    
    Args:
        df: Input dataframe with raw features
        categorical_cols: List of categorical column names to one-hot encode
        scaler: Fitted StandardScaler object (for LogisticRegression)
        lr_feature_cols: Expected feature columns after encoding (for alignment)
    
    Returns:
        Processed dataframe ready for model prediction
    """
    df_proc = df.copy()
    
    # Step 1: Create missing value indicators
    df_proc["missing_any"] = df_proc.isnull().any(axis=1).astype(int)
    logger.info("Step 1: Created missing value indicators")
    
    # Step 2: Fill missing values
    df_proc.fillna(0, inplace=True)
    logger.info("Step 2: Filled missing values with 0")
    
    # Step 3: One-hot encode categorical features (if model requires it)
    if categorical_cols is not None and len(categorical_cols) > 0:
        df_proc = pd.get_dummies(
            df_proc, 
            columns=categorical_cols, 
            drop_first=True, 
            dtype=int
        )
        logger.info(f"Step 3: One-hot encoded {len(categorical_cols)} categorical columns")
    
    # Step 4: Align columns with training feature set (if provided)
    if lr_feature_cols is not None:
        # Add missing columns with 0s
        for col in lr_feature_cols:
            if col not in df_proc.columns:
                df_proc[col] = 0
        
        # Keep only the expected columns in the right order
        df_proc = df_proc[lr_feature_cols]
        logger.info(f"Step 3b: Aligned features to {len(lr_feature_cols)} columns")
    
    # Step 4: Scale features (if scaler provided - for LogisticRegression)
    if scaler is not None:
        # Identify numeric columns to scale (exclude binary flags and dummies)
        numeric_cols = [c for c in df_proc.columns 
                       if not c.endswith("_flag") 
                       and df_proc[c].dtype in ['float64', 'int64']
                       and df_proc[c].nunique() > 2]  # Exclude binary columns
        
        if numeric_cols:
            df_proc[numeric_cols] = scaler.transform(df_proc[numeric_cols])
            logger.info(f"Step 4: Scaled {len(numeric_cols)} numeric features")
    
    return df_proc


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

    # Extract raw feature columns (same 12 features as training)
    feature_cols = ['tenure_days_at_snapshot', 'registered_via', 'city_clean', 
                'sum_secs_w30', 'active_days_w30', 'complete_rate_w30', 
                'sum_secs_w7', 'engagement_ratio_7_30', 'days_since_last_play', 
                'trend_secs_w30', 'auto_renew_share', 'last_is_auto_renew']
    X_inference = features_pdf[feature_cols].copy()

    # Load MLflow model
    model = load_mlflow_model(model_name)
    
    # Load preprocessing artifacts
    model_uri = f"models:/{model_name}/Production" if not model_name.startswith("models:/") else model_name
    scaler, lr_feature_cols, categorical_cols = load_preprocessing_artifacts(model_uri)
    
    # Apply preprocessing (Steps 1-4: missing indicators, fill, one-hot encode, scale)
    logger.info("Applying preprocessing steps...")
    X_processed = apply_preprocessing(
        X_inference, 
        categorical_cols=categorical_cols,
        scaler=scaler,
        lr_feature_cols=lr_feature_cols
    )
    logger.info(f"Processed features shape: {X_processed.shape}")

    # Predict
    y_proba = model.predict_proba(X_processed)[:, 1]

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