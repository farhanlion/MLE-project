#!/usr/bin/env python3
# coding: utf-8
"""
preprocessing_before_fit.py

Shared preprocessing + data-splitting utilities used by the 3 model training scripts.
Exports:
- prepare_data_for_training(args): returns dict of X/y splits, scaler, lr feature columns, class weights, cv object
- build_config(...) and a few spark helpers are re-used internally.
"""

import os
import logging
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


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

    cfg["data_start_date"] = cfg["train_start"]
    cfg["data_end_date"]   = cfg["oot_end"]

    return cfg


def start_spark(app_name="model_training", master="local[*]"):
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


def prepare_lr_df(df, categorical_cols, train_cols=None, scaler=None, fit=False):
    df_proc = df.copy()
    if categorical_cols:
        df_proc = pd.get_dummies(df_proc, columns=categorical_cols, drop_first=True, dtype=int)
    if train_cols is None:
        train_cols = df_proc.columns.tolist()
    # ensure same columns in this order
    for colname in train_cols:
        if colname not in df_proc.columns:
            df_proc[colname] = 0
    df_proc = df_proc[train_cols]
    # scale numeric (exclude dummies)
    # compute numeric columns as those not in categorical original list and not all 0/1
    numeric_cols = [c for c in df_proc.columns if c not in train_cols or True]  # placeholder
    # we instead detect numeric by dtype:
    numeric_cols = [c for c in df_proc.columns if df_proc[c].dtype in [float, int] and not c.startswith(tuple(categorical_cols))]
    if fit and scaler is not None:
        scaler.fit(df_proc[numeric_cols])
    if scaler is not None:
        df_proc[numeric_cols] = scaler.transform(df_proc[numeric_cols])
    return df_proc, train_cols, scaler


def prepare_data_for_training(args):
    """
    Reads parquet features & labels, applies date windows, stratified sampling (if requested),
    converts to pandas, prepares LR and tree datasets, computes scaler, lr columns and class weights,
    and returns them in a dict.
    """
    cfg = build_config(args.train_date, train_months=args.train_months,
                       val_months=args.val_months, test_months=args.test_months, oot_months=args.oot_months)

    logger.info("Config date windows computed")
    logger.info(f"Train: {cfg['train_start'].date()} → {cfg['train_end'].date()}")
    logger.info(f"Val:   {cfg['val_start'].date()} → {cfg['val_end'].date()}")
    logger.info(f"Test:  {cfg['test_start'].date()} → {cfg['test_end'].date()}")
    logger.info(f"OOT:   {cfg['oot_start'].date()} → {cfg['oot_end'].date()}")

    spark = start_spark(app_name="kkbox_preprocessing")
    features_store_sdf = spark.read.parquet(args.features_path)
    labels_store_sdf   = spark.read.parquet(args.labels_path)

    # apply global filter
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

    joined_sdf = features_sdf.join(labels_sdf, on=["msno", "snapshot_date"], how="inner")
    logger.info(f"Joined rows: {joined_sdf.count()}")

    label_col = args.label_col
    if label_col not in joined_sdf.columns:
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

    # Create split SDFs
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
    logger.info("Stopped Spark - continuing in pandas")

    # Hardcoded feature list (same as original)
    feature_cols = [
        'tenure_days_at_snapshot', 'registered_via', 'city_clean',
        'sum_secs_w30', 'active_days_w30', 'complete_rate_w30',
        'sum_secs_w7', 'engagement_ratio_7_30', 'days_since_last_play',
        'trend_secs_w30', 'auto_renew_share', 'last_is_auto_renew'
    ]
    categorical_cols = ['registered_via', 'city_clean']

    # Build X/y
    X_train = sampled_pdfs["train"][feature_cols].copy()
    y_train = sampled_pdfs["train"][label_col].astype(int).copy()

    X_val = sampled_pdfs["val"][feature_cols].copy()
    y_val = sampled_pdfs["val"][label_col].astype(int).copy()

    X_test = sampled_pdfs["test"][feature_cols].copy()
    y_test = sampled_pdfs["test"][label_col].astype(int).copy()

    X_oot = sampled_pdfs["oot"][feature_cols].copy()
    y_oot = sampled_pdfs["oot"][label_col].astype(int).copy()

    # missing indicators + fill
    for df in [X_train, X_val, X_test, X_oot]:
        df["missing_any"] = df.isnull().any(axis=1).astype(int)
        df.fillna(0, inplace=True)

    # prepare LR inputs (ohe + scaler)
    scaler = StandardScaler()
    X_train_lr = pd.get_dummies(X_train.copy(), columns=categorical_cols, drop_first=True, dtype=int)
    lr_cols = X_train_lr.columns.tolist()
    numeric_cols = [c for c in X_train_lr.columns if X_train_lr[c].dtype in [float, int]]
    scaler.fit(X_train_lr[numeric_cols])
    X_train_lr[numeric_cols] = scaler.transform(X_train_lr[numeric_cols])

    def apply_lr_prep(df):
        df_proc = pd.get_dummies(df.copy(), columns=categorical_cols, drop_first=True, dtype=int)
        for c in lr_cols:
            if c not in df_proc.columns:
                df_proc[c] = 0
        df_proc = df_proc[lr_cols]
        df_proc[numeric_cols] = scaler.transform(df_proc[numeric_cols])
        return df_proc

    X_val_lr = apply_lr_prep(X_val)
    X_test_lr = apply_lr_prep(X_test)
    X_oot_lr = apply_lr_prep(X_oot)

    # tree datasets: raw (no scaling / ohe)
    X_train_tree = X_train.copy()
    X_val_tree = X_val.copy()
    X_test_tree = X_test.copy()
    X_oot_tree = X_oot.copy()

    # class weights
    cw_vals = compute_class_weight(class_weight="balanced", classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: cw_vals[0], 1: cw_vals[1]}
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    # cross-val
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.random_state)

    return {
        "X_train_lr": X_train_lr, "y_train": y_train,
        "X_val_lr": X_val_lr, "y_val": y_val,
        "X_test_lr": X_test_lr, "y_test": y_test,
        "X_oot_lr": X_oot_lr, "y_oot": y_oot,
        "X_train_tree": X_train_tree, "X_val_tree": X_val_tree,
        "X_test_tree": X_test_tree, "X_oot_tree": X_oot_tree,
        "scaler": scaler, "lr_cols": lr_cols, "categorical_cols": categorical_cols,
        "class_weight_dict": class_weight_dict, "scale_pos_weight": scale_pos, "cv": cv
    }
