#!/usr/bin/env python3
# coding: utf-8
"""
Train XGBoost. Reading + splits done here. Calls preprocessing_before_fit only for feature transforms
(we use preprocess_features_for_tree which does minimal filling).
"""

import argparse
import logging
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, randint

    
sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))

from model_preprocessor import preprocess_features_for_tree


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# reuse helper functions from train_lr (build_config, start_spark, etc.)
def build_config(train_date_str, train_months=8, val_months=2, test_months=2, oot_months=2):
    from datetime import datetime, timedelta
    from dateutil.relativedelta import relativedelta
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
    cfg["data_start_date"] = cfg["train_start"]
    cfg["data_end_date"]   = cfg["oot_end"]
    return cfg

def start_spark(app_name="model_training", master="local[*]"):   
    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        .config("spark.driver.memory", "8g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

def stratified_sample_spark(sdf, label_col, frac, seed=42):
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

def evaluate_model(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y, proba),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0)
    }
    return metrics

def main(args):
    cfg = build_config(args.train_date, train_months=args.train_months,
                       val_months=args.val_months, test_months=args.test_months, oot_months=args.oot_months)
    logger.info(f"Train: {cfg['train_start'].date()} → {cfg['train_end'].date()}")

    spark = start_spark(app_name="kkbox_xgb_train")
    features_store_sdf = spark.read.parquet(args.features_path)
    labels_store_sdf   = spark.read.parquet(args.labels_path)

    features_sdf = features_store_sdf.filter(
        (col("snapshot_date") >= F.lit(cfg["data_start_date"].date())) &
        (col("snapshot_date") <= F.lit(cfg["data_end_date"].date()))
    )
    labels_sdf = labels_store_sdf.filter(
        (col("snapshot_date") >= F.lit(cfg["data_start_date"].date())) &
        (col("snapshot_date") <= F.lit(cfg["data_end_date"].date()))
    )

    joined = features_sdf.join(labels_sdf, on=["msno", "snapshot_date"], how="inner")

    label_col = args.label_col
    if label_col not in joined.columns:
        fallback = None
        for c in ["label", "is_churn", "churn"]:
            if c in joined.columns:
                fallback = c; break
        if fallback is None:
            raise ValueError(f"Label column '{label_col}' not found.")
        label_col = fallback

    splits = {}
    splits["train"] = joined.filter((col("snapshot_date") >= F.lit(cfg["train_start"].date())) &
                                   (col("snapshot_date") <= F.lit(cfg["train_end"].date())))
    splits["val"] = joined.filter((col("snapshot_date") >= F.lit(cfg["val_start"].date())) &
                                 (col("snapshot_date") <= F.lit(cfg["val_end"].date())))
    splits["test"] = joined.filter((col("snapshot_date") >= F.lit(cfg["test_start"].date())) &
                                  (col("snapshot_date") <= F.lit(cfg["test_end"].date())))
    splits["oot"] = joined.filter((col("snapshot_date") >= F.lit(cfg["oot_start"].date())) &
                                 (col("snapshot_date") <= F.lit(cfg["oot_end"].date())))

    sampled_pdfs = {}
    for k, sdf in splits.items():
        if args.sample_frac < 1.0:
            sdf_cast = sdf.withColumn(label_col, col(label_col).cast("int"))
            sampled = stratified_sample_spark(sdf_cast, label_col, args.sample_frac, seed=args.random_state)
        else:
            sampled = sdf
        pdf = sdf_to_pandas(sampled)
        sampled_pdfs[k] = pdf

    spark.stop()
    logger.info("Stopped Spark")

    feature_cols = [
        'tenure_days_at_snapshot', 'registered_via', 'city_clean',
        'sum_secs_w30', 'active_days_w30', 'complete_rate_w30',
        'sum_secs_w7', 'engagement_ratio_7_30', 'days_since_last_play',
        'trend_secs_w30', 'auto_renew_share', 'last_is_auto_renew'
    ]

    X_train = sampled_pdfs["train"][feature_cols].copy()
    y_train = sampled_pdfs["train"][label_col].astype(int).copy()
    X_val = sampled_pdfs["val"][feature_cols].copy()
    y_val = sampled_pdfs["val"][label_col].astype(int).copy()
    X_test = sampled_pdfs["test"][feature_cols].copy()
    y_test = sampled_pdfs["test"][label_col].astype(int).copy()
    X_oot = sampled_pdfs["oot"][feature_cols].copy()
    y_oot = sampled_pdfs["oot"][label_col].astype(int).copy()

    # minimal preprocessing for trees: missing flag + fill
    X_train_tree, X_val_tree, X_test_tree, X_oot_tree = preprocess_features_for_tree(
        [X_train, X_val, X_test, X_oot]
    )

    # xgboost setup
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_base = XGBClassifier(scale_pos_weight=scale_pos, random_state=args.random_state,
                             use_label_encoder=False, eval_metric="auc", n_jobs=-1)

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

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)

    search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=xgb_param_dist,
        n_iter=args.n_iter,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=args.random_state,
        return_train_score=True
    )

    with mlflow.start_run(run_name="XGBoost_tuning"):
        search.fit(X_train_tree, y_train)
        best = search.best_estimator_
        mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
        for k, v in search.best_params_.items():
            mlflow.log_param(f"best_{k}", v)

        for name, X, y in [("train", X_train_tree, y_train), ("val", X_val_tree, y_val),
                           ("test", X_test_tree, y_test), ("oot", X_oot_tree, y_oot)]:
            proba = best.predict_proba(X)[:,1]
            preds = (proba >= 0.5).astype(int)
            mlflow.log_metric(f"{name}_roc_auc", roc_auc_score(y, proba))
            mlflow.log_metric(f"{name}_precision", precision_score(y, preds, zero_division=0))
            mlflow.log_metric(f"{name}_recall", recall_score(y, preds, zero_division=0))
            mlflow.log_metric(f"{name}_f1", f1_score(y, preds, zero_division=0))

        artifact_path = "XGBoost_model"
        mlflow.sklearn.log_model(best, artifact_path)

    logger.info("XGBoost training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_date", type=str, required=True)
    parser.add_argument("--features_path", type=str, default="/app/datamart/gold/feature_store/")
    parser.add_argument("--labels_path", type=str, default="/app/datamart/gold/label_store/")
    parser.add_argument("--sample_frac", type=float, default=0.3)
    parser.add_argument("--label_col", type=str, default="label")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="http://mlflow:5000")
    parser.add_argument("--mlflow_experiment", type=str, default="kkbox-churn-prediction")
    parser.add_argument("--n_iter", type=int, default=10)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--train_months", type=int, default=8)
    parser.add_argument("--val_months", type=int, default=2)
    parser.add_argument("--test_months", type=int, default=2)
    parser.add_argument("--oot_months", type=int, default=2)
    args = parser.parse_args()
    main(args)
