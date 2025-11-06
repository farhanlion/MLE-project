#!/usr/bin/env python3
# coding: utf-8
"""
model_training_spark.py

- Reads features and labels from parquet via Spark
- Filters by a date window computed from --train_date and periods
- Joins features and labels by msno + snapshot_date
- Splits into train/val/test/oot using time-based windows
- Performs stratified sampling per segment (sample_frac)
- Converts sampled splits to pandas
- Trains Logistic Regression, XGBoost, RandomForest with RandomizedSearchCV
- Logs runs and artifacts to MLflow and registers best model for each algorithm
"""

import argparse
import os
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from scipy.stats import uniform, randint

# -------------------------
# Logging
# -------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# -------------------------
# Utility functions
# -------------------------
def build_config(train_date_str, train_months=8, val_months=2, test_months=2, oot_months=2):
    model_train_date = datetime.strptime(train_date_str, "%Y-%m-%d")
    cfg = {}
    cfg["model_train_date"] = model_train_date

    cfg["oot_end"] = model_train_date - timedelta(days=1)
    cfg["oot_start"] = model_train_date - relativedelta(months=oot_months)

    cfg["test_end"] = cfg["oot_start"] - timedelta(days=1)
    cfg["test_start"] = cfg["oot_start"] - relativedelta(months=test_months)

    cfg["val_end"] = cfg["test_start"] - timedelta(days=1)
    cfg["val_start"] = cfg["test_start"] - relativedelta(months=val_months)

    cfg["train_end"] = cfg["val_start"] - timedelta(days=1)
    cfg["train_start"] = cfg["val_start"] - relativedelta(months=train_months)

    # global data window to read from parquet (min / max)
    cfg["data_start_date"] = cfg["train_start"]
    cfg["data_end_date"]   = cfg["oot_end"]

    return cfg

def start_spark(app_name="model_training", master="local[*]"):
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def read_and_filter_parquet(spark, path, start_date, end_date):
    """
    Read parquet and filter snapshot_date between start_date and end_date (inclusive).
    start_date/end_date are Python datetime objects -> we compare on date part.
    """
    logger.info(f"Reading parquet from {path} and filtering {start_date.date()} → {end_date.date()}")
    sdf = (
        spark.read.parquet(path)
             .filter((col("snapshot_date") >= F.lit(start_date.date())) &
                     (col("snapshot_date") <= F.lit(end_date.date())))
    )
    logger.info(f"Read rows: {sdf.count()}")
    return sdf

def join_features_labels(features_sdf, labels_sdf):
    joined = features_sdf.join(labels_sdf, on=["msno", "snapshot_date"], how="inner")
    logger.info(f"Joined rows: {joined.count()}")
    return joined

def stratified_sample_spark(sdf, label_col, frac, seed=42):
    """
    Perform stratified sampling on a Spark DataFrame.
    `frac` is global fraction for each label value.
    For binary labels, we build fractions dict {0: frac, 1: frac}
    """
    values = [r[label_col] for r in sdf.select(label_col).distinct().collect()]
    fractions = {int(v): float(frac) for v in values}
    logger.info(f"Sampling fractions per label: {fractions}")
    sampled = sdf.sampleBy(label_col, fractions, seed)
    logger.info(f"Sampled rows: {sampled.count()}")
    return sampled

def sdf_to_pandas(sdf):
    pdf = sdf.toPandas()
    logger.info(f"Converted Spark DF to pandas: {pdf.shape}")
    return pdf

# -------------------------
# Modeling helpers
# -------------------------
def evaluate_model(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y, proba),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0)
    }
    return metrics, proba, preds

def log_metrics_mlflow(metrics_dict):
    for k, v in metrics_dict.items():
        mlflow.log_metric(k, v)

# -------------------------
# Train & Log function (registers model)
# -------------------------
def train_and_register(
    model_name, estimator, param_dist,
    X_train, y_train, X_val, y_val, X_test, y_test, X_oot, y_oot,
    n_iter, cv_strategy, random_state=42, scaler=None, lr_cols=None, categorical_cols=None
):
    logger.info(f"Starting training for {model_name}")

    with mlflow.start_run(run_name=f"{model_name}_tuning"):
        mlflow.set_tag("model_type", model_name)
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("cv_folds", getattr(cv_strategy, "n_splits", cv_strategy))
        mlflow.log_param("random_state", random_state)

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv_strategy,
            n_jobs=-1,
            random_state=random_state,
            return_train_score=True
        )

        search.fit(X_train, y_train)

        best = search.best_estimator_
        logger.info(f"Best CV roc_auc: {search.best_score_:.4f}")
        mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))

        # log best params
        for k, v in search.best_params_.items():
            mlflow.log_param(f"best_{k}", v)

        # Evaluate across datasets
        metrics_agg = {}
        for label, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
            ("oot", X_oot, y_oot),
        ]:
            m, proba, preds = evaluate_model(best, X, y)
            # prefix metrics with dataset name
            m_prefixed = {f"{label}_{mk}": mv for mk, mv in m.items()}
            metrics_agg.update(m_prefixed)
            log_metrics_mlflow(m_prefixed)

        # Log model artifact
        artifact_path = f"{model_name}_model"
        mlflow.sklearn.log_model(best, artifact_path)
        
        # Save preprocessing artifacts for Logistic Regression
        if scaler is not None:
            # Save scaler
            scaler_path = "scaler.pkl"
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
            os.remove(scaler_path)
            logger.info("Saved scaler to MLflow")
            
        if lr_cols is not None:
            # Save feature column names after one-hot encoding
            lr_cols_path = "lr_feature_columns.txt"
            with open(lr_cols_path, 'w') as f:
                f.write('\n'.join(lr_cols))
            mlflow.log_artifact(lr_cols_path, artifact_path="preprocessing")
            os.remove(lr_cols_path)
            logger.info(f"Saved {len(lr_cols)} LR feature columns to MLflow")
            
        if categorical_cols is not None and len(categorical_cols) > 0:
            # Save categorical column names
            cat_cols_path = "categorical_columns.txt"
            with open(cat_cols_path, 'w') as f:
                f.write('\n'.join(categorical_cols))
            mlflow.log_artifact(cat_cols_path, artifact_path="preprocessing")
            os.remove(cat_cols_path)
            logger.info(f"Saved {len(categorical_cols)} categorical columns to MLflow")

        # Register the model in Model Registry
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        try:
            reg_result = mlflow.register_model(model_uri=model_uri, name=model_name)
            logger.info(f"Registered model: {model_name} version={reg_result.version}")
            # log model registry info
            mlflow.set_tag("registered_model_name", model_name)
            mlflow.log_param("registered_model_version", reg_result.version)
        except Exception as e:
            logger.warning(f"Model registration failed for {model_name}: {e}")

        logger.info(f"Completed training & register for {model_name}")
        return best, metrics_agg

# -------------------------
# Main pipeline
# -------------------------
def main(args):
    # Build config windows
    cfg = build_config(args.train_date, train_months=args.train_months,
                       val_months=args.val_months, test_months=args.test_months, oot_months=args.oot_months)

    logger.info("Config date windows computed")
    logger.info(f"Train: {cfg['train_start'].date()} → {cfg['train_end'].date()}")
    logger.info(f"Val:   {cfg['val_start'].date()} → {cfg['val_end'].date()}")
    logger.info(f"Test:  {cfg['test_start'].date()} → {cfg['test_end'].date()}")
    logger.info(f"OOT:   {cfg['oot_start'].date()} → {cfg['oot_end'].date()}")

    # Start Spark
    spark = start_spark(app_name="kkbox_model_training")

    # Read feature and label stores (only once), filtered to data_start/date_end
    features_store_sdf = spark.read.parquet(args.features_path)
    labels_store_sdf   = spark.read.parquet(args.labels_path)

    logger.info("Loaded raw parquet datasets (no filter yet)")

    # Apply date window filter in one chain (compact)
    features_sdf = (
        features_store_sdf
        .filter((col("snapshot_date") >= F.lit(cfg["data_start_date"].date())) &
                (col("snapshot_date") <= F.lit(cfg["data_end_date"].date())))
    )
    labels_sdf = (
        labels_store_sdf
        .filter((col("snapshot_date") >= F.lit(cfg["data_start_date"].date())) &
                (col("snapshot_date") <= F.lit(cfg["data_end_date"].date())))
    )

    logger.info(f"Filtered features rows: {features_sdf.count()}")
    logger.info(f"Filtered labels rows: {labels_sdf.count()}")

    # Join
    joined_sdf = join_features_labels(features_sdf, labels_sdf)

    # Determine label column
    label_col = args.label_col
    if label_col not in joined_sdf.columns:
        # try fallback common names
        fallback = None
        for candidate in ["label", "is_churn", "churn"]:
            if candidate in joined_sdf.columns:
                fallback = candidate
                break
        if fallback is None:
            raise ValueError(f"Label column '{label_col}' not present in joined dataset and no fallback found.")
        else:
            label_col = fallback
            logger.warning(f"Using fallback label column: {label_col}")

    # Split into train/val/test/oot by snapshot_date
    # We'll filter joined_sdf by those windows to produce each split
    splits = {}
    splits["train"] = joined_sdf.filter(
        (col("snapshot_date") >= F.lit(cfg["train_start"].date())) &
        (col("snapshot_date") <= F.lit(cfg["train_end"].date()))
    )
    splits["val"] = joined_sdf.filter(
        (col("snapshot_date") >= F.lit(cfg["val_start"].date())) &
        (col("snapshot_date") <= F.lit(cfg["val_end"].date()))
    )
    splits["test"] = joined_sdf.filter(
        (col("snapshot_date") >= F.lit(cfg["test_start"].date())) &
        (col("snapshot_date") <= F.lit(cfg["test_end"].date()))
    )
    splits["oot"] = joined_sdf.filter(
        (col("snapshot_date") >= F.lit(cfg["oot_start"].date())) &
        (col("snapshot_date") <= F.lit(cfg["oot_end"].date()))
    )

    for k in splits:
        logger.info(f"{k} rows before sampling: {splits[k].count()}")

    # Perform stratified sampling per split if sample_frac < 1.0
    sampled_pdfs = {}
    for k, sdf in splits.items():
        if args.sample_frac < 1.0:
            # ensure label column type int
            sdf_cast = sdf.withColumn(label_col, col(label_col).cast("int"))
            sampled = stratified_sample_spark(sdf_cast, label_col, args.sample_frac, seed=args.random_state)
        else:
            sampled = sdf
        # Convert to pandas
        pdf = sdf_to_pandas(sampled)
        sampled_pdfs[k] = pdf

    # Close spark (we have pandas)
    spark.stop()
    logger.info("Stopped Spark - continuing in pandas")
    
    # Use hardcoded feature list
    feature_cols = [
        'tenure_days_at_snapshot', 'registered_via', 'city_clean', 
        'sum_secs_w30', 'active_days_w30', 'complete_rate_w30', 
        'sum_secs_w7', 'engagement_ratio_7_30', 'days_since_last_play', 
        'trend_secs_w30', 'auto_renew_share', 'last_is_auto_renew'
    ]

    logger.info(f"Using {len(feature_cols)} hardcoded feature columns")

    # Create X/y for each split
    X_train = sampled_pdfs["train"][feature_cols].copy()
    y_train = sampled_pdfs["train"][label_col].astype(int).copy()

    X_val = sampled_pdfs["val"][feature_cols].copy()
    y_val = sampled_pdfs["val"][label_col].astype(int).copy()

    X_test = sampled_pdfs["test"][feature_cols].copy()
    y_test = sampled_pdfs["test"][label_col].astype(int).copy()

    X_oot = sampled_pdfs["oot"][feature_cols].copy()
    y_oot = sampled_pdfs["oot"][label_col].astype(int).copy()

    # Preprocessing: missing-value indicators + fill
    for df in [X_train, X_val, X_test, X_oot]:
        # example indicators — adapt to your data
        # create generic missing flag: any feature missing
        df["missing_any"] = df.isnull().any(axis=1).astype(int)
        df.fillna(0, inplace=True)

    # Prepare for LR: OHE categorical if any
    # Hardcode categorical columns that need one-hot encoding
    categorical_cols = ['registered_via', 'city_clean']
    logger.info(f"Using hardcoded categorical columns for one-hot encoding: {categorical_cols}")

    # Build LR datasets (one-hot + scaler)
    def prepare_lr_df(df, train_cols=None, scaler=None, fit=False):
        df_proc = df.copy()
        if categorical_cols:
            df_proc = pd.get_dummies(df_proc, columns=categorical_cols, drop_first=True, dtype=int)
        if train_cols is None:
            train_cols = df_proc.columns
        for colname in train_cols:
            if colname not in df_proc.columns:
                df_proc[colname] = 0
        df_proc = df_proc[train_cols]
        # scale numeric (exclude dummies and binary flags)
        numeric_cols = [c for c in df_proc.columns if c not in categorical_cols and not c.endswith("_flag") and df_proc[c].dtype in [float, int]]
        # remove boolean like columns that are actually binary dummies: user may further refine
        if fit:
            scaler.fit(df_proc[numeric_cols])
        if scaler:
            df_proc[numeric_cols] = scaler.transform(df_proc[numeric_cols])
        return df_proc, train_cols, scaler

    scaler = StandardScaler()
    X_train_lr, lr_cols, scaler = prepare_lr_df(X_train, train_cols=None, scaler=scaler, fit=True)
    X_val_lr, _, _ = prepare_lr_df(X_val, train_cols=lr_cols, scaler=scaler, fit=False)
    X_test_lr, _, _ = prepare_lr_df(X_test, train_cols=lr_cols, scaler=scaler, fit=False)
    X_oot_lr, _, _ = prepare_lr_df(X_oot, train_cols=lr_cols, scaler=scaler, fit=False)

    # Prepare tree datasets (no scaling)
    X_train_tree = X_train.copy()
    X_val_tree = X_val.copy()
    X_test_tree = X_test.copy()
    X_oot_tree = X_oot.copy()

    # Compute class weights
    cw_vals = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=y_train)
    class_weight_dict = {0: cw_vals[0], 1: cw_vals[1]}

    # Prepare hyperparameter spaces and CV
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)
    N_ITER = args.n_iter

    lr_param_dist = {
        'C': uniform(0.01, 10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    }

    xgb_param_dist = {
        'max_depth': randint(3, 10),
        'learning_rate': uniform(0.01, 0.3),
        'n_estimators': randint(100, 300),
        'min_child_weight': randint(1, 10),
        'subsample': uniform(0.6, 0.4),
        'colsample_bytree': uniform(0.6, 0.4),
        'gamma': uniform(0, 5),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 1)
    }

    rf_param_dist = {
        'n_estimators': randint(100, 300),
        'max_depth': [None] + list(range(5, 25, 5)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }

    # Initialize models
    lr_base = LogisticRegression(class_weight=class_weight_dict, random_state=args.random_state, n_jobs=-1)
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_base = XGBClassifier(scale_pos_weight=scale_pos, random_state=args.random_state, use_label_encoder=False, eval_metric="auc", n_jobs=-1)
    rf_base = RandomForestClassifier(class_weight=class_weight_dict, random_state=args.random_state, n_jobs=-1)

    # Initialize mlflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    # Train & register each model
    lr_model, lr_metrics = train_and_register(
        "LogisticRegression", lr_base, lr_param_dist,
        X_train_lr, y_train, X_val_lr, y_val, X_test_lr, y_test, X_oot_lr, y_oot,
        n_iter=N_ITER, cv_strategy=cv, random_state=args.random_state,
        scaler=scaler, lr_cols=list(lr_cols), categorical_cols=categorical_cols
    )

    xgb_model, xgb_metrics = train_and_register(
        "XGBoost", xgb_base, xgb_param_dist,
        X_train_tree, y_train, X_val_tree, y_val, X_test_tree, y_test, X_oot_tree, y_oot,
        n_iter=N_ITER, cv_strategy=cv, random_state=args.random_state
    )

    rf_model, rf_metrics = train_and_register(
        "RandomForest", rf_base, rf_param_dist,
        X_train_tree, y_train, X_val_tree, y_val, X_test_tree, y_test, X_oot_tree, y_oot,
        n_iter=N_ITER, cv_strategy=cv, random_state=args.random_state
    )

    # Summarize (simple)
    summary = pd.DataFrame({
        "Model": ["LogisticRegression", "XGBoost", "RandomForest"],
        "Val_ROC_AUC": [lr_metrics.get("val_roc_auc"), xgb_metrics.get("val_roc_auc"), rf_metrics.get("val_roc_auc")],
        "Test_ROC_AUC": [lr_metrics.get("test_roc_auc"), xgb_metrics.get("test_roc_auc"), rf_metrics.get("test_roc_auc")],
        "OOT_ROC_AUC": [lr_metrics.get("oot_roc_auc"), xgb_metrics.get("oot_roc_auc"), rf_metrics.get("oot_roc_auc")]
    })
    logger.info("Model comparison summary:")
    logger.info("\n" + summary.to_string(index=False))

    logger.info("Training script completed.")


# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KKBox churn model training with Spark ingestion and MLflow")
    parser.add_argument("--train_date", type=str, required=True, help="model training cutoff date YYYY-MM-DD")
    parser.add_argument("--features_path", type=str, default="/app/datamart/gold/feature_store/", help="parquet path for features")
    parser.add_argument("--labels_path", type=str, default="/app/datamart/gold/label_store/", help="parquet path for labels")
    parser.add_argument("--sample_frac", type=float, default=0.3, help="stratified sample fraction per split (0..1)")
    parser.add_argument("--feature_prefix", type=str, default="fe_", help="prefix to detect feature columns")
    parser.add_argument("--label_col", type=str, default="label", help="label column name in labels parquet")
    parser.add_argument("--categorical_cols", type=str, default="", help="comma-separated categorical columns (optional)")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="http://mlflow:5000", help="MLflow tracking URI")
    parser.add_argument("--mlflow_experiment", type=str, default="kkbox-churn-prediction", help="MLflow experiment name")
    parser.add_argument("--n_iter", type=int, default=10, help="RandomizedSearchCV n_iter")
    parser.add_argument("--cv_folds", type=int, default=5, help="CV folds")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed")
    parser.add_argument("--train_months", type=int, default=8)
    parser.add_argument("--val_months", type=int, default=2)
    parser.add_argument("--test_months", type=int, default=2)
    parser.add_argument("--oot_months", type=int, default=2)

    args = parser.parse_args()
    main(args)