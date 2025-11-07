#!/usr/bin/env python3
# monitor_predictions.py
"""
Comprehensive monitoring for predictions.

Example:
python monitor_predictions.py \
  --model_label 2ddc6524115b4d0ebd2d3161c7081289 \
  --snapshotdate 2016-01-01 \
  --baseline_date 2015-12-01 \
  --output_dir datamart/gold/model_monitoring/2ddc... \
  --labels_path /app/datamart/gold/labels/labels.parquet \
  --log_mlflow

Outputs:
 - {output_dir}/monitor_report_{snapshotdate}.json
 - {output_dir}/monitor_table_{snapshotdate}.csv
"""

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime
import math

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import pyspark
from pyspark.sql.functions import col
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

# Optional MLflow logging
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("monitor_predictions")


def find_prediction_path(model_label, snapshotdate):
    base_dir = Path("datamart/gold/model_predictions") / str(model_label)
    name = f"{model_label}_predictions_{snapshotdate.replace('-', '_')}.parquet"
    p = base_dir / name
    if p.exists():
        return str(p)
    # wildcard lookup
    matches = list(base_dir.glob(f"*{snapshotdate.replace('-', '_')}*.parquet"))
    if matches:
        return str(matches[0])
    raise FileNotFoundError(f"No prediction file for {model_label} @ {snapshotdate}")


def read_preds(spark, path):
    """
    Read parquet at `path` and return a pandas DataFrame with a column named
    'prediction_proba'.  Tries several common column-name patterns and falls
    back to picking the first numeric column (excluding id/date).
    """
    sdf = spark.read.parquet(path)
    pdf = sdf.toPandas()

    # canonicalize column names for searching
    cols = pdf.columns.tolist()
    cols_lower = [c.lower() for c in cols]

    # priority search patterns (in order)
    preferred = [
        "prediction_proba", "model_prediction_proba", "churn_proba", "proba", "probability",
        "prob", "score", "prediction", "pred"
    ]

    chosen = None
    for p in preferred:
        # try exact match first (case-insensitive)
        for i, lc in enumerate(cols_lower):
            if lc == p:
                chosen = cols[i]
                break
        if chosen:
            break
        # then substring match
        for i, lc in enumerate(cols_lower):
            if p in lc:
                chosen = cols[i]
                break
        if chosen:
            break

    # If still not found, pick the first numeric column excluding id/date
    if chosen is None:
        # exclude identifier/date-ish columns
        exclude = set(["msno", "snapshot_date", "id", "member_id", "userid"])
        numeric_cols = [c for c in cols if c not in exclude and pd.api.types.is_numeric_dtype(pdf[c])]
        if len(numeric_cols) == 0:
            raise KeyError(
                "No prediction-like column found. Searched for: "
                + ", ".join(preferred)
                + ". Also no numeric columns available to fall back to."
            )
        chosen = numeric_cols[0]  # pick first numeric column

    # rename chosen column to 'prediction_proba' if needed
    if chosen != "prediction_proba":
        pdf = pdf.rename(columns={chosen: "prediction_proba"})

    # sanity: ensure it's numeric
    pdf["prediction_proba"] = pd.to_numeric(pdf["prediction_proba"], errors="coerce")
    if pdf["prediction_proba"].isnull().all():
        raise ValueError(f"Chosen prediction column '{chosen}' could not be converted to numeric.")

    return pdf



def psi(expected, actual, bins=10, eps=1e-8):
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    # create breaks from expected quantiles
    try:
        quantiles = np.linspace(0, 1, bins + 1)
        breaks = np.unique(np.quantile(expected, quantiles))
        if len(breaks) <= 1:
            breaks = np.linspace(min(expected), max(expected), bins + 1)
    except Exception:
        breaks = np.linspace(min(expected), max(expected), bins + 1)

    exp_counts, _ = np.histogram(expected, bins=breaks)
    act_counts, _ = np.histogram(actual, bins=breaks)
    exp_props = exp_counts / (exp_counts.sum() + eps)
    act_props = act_counts / (act_counts.sum() + eps)

    # avoid zeros
    exp_props = np.where(exp_props == 0, eps, exp_props)
    act_props = np.where(act_props == 0, eps, act_props)

    psi_val = np.sum((exp_props - act_props) * np.log(exp_props / act_props))
    return float(psi_val)


def summarize(arr):
    arr = np.array(arr)
    if arr.size == 0:
        return {}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(float(np.std(arr))),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "deciles": {f"p{int(q*100)}": float(np.quantile(arr, q)) for q in np.linspace(0, 1, 11)}
    }


def compute_metrics_with_labels(pdf_preds, labels_pdf, label_col="label"):
    # join on msno + snapshot_date
    merged = pdf_preds.merge(labels_pdf, on=["msno", "snapshot_date"], how="left")
    if label_col not in merged.columns:
        logger.warning(f"label column {label_col} not found in labels file")
        return None
    y_true = merged[label_col].astype(int)
    y_score = merged["prediction_proba"].astype(float)

    metrics = {}
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["roc_auc"] = None
    preds = (y_score >= 0.5).astype(int)
    metrics["precision"] = float(precision_score(y_true, preds, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, preds, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, preds, zero_division=0))
    return metrics


def main(args):
    spark = pyspark.sql.SparkSession.builder.master("local[*]").appName("monitor_predictions").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    # resolve paths
    current_path = args.pred_path or find_prediction_path(args.model_label, args.snapshotdate)
    baseline_path = args.baseline_path or find_prediction_path(args.model_label, args.baseline_date)

    logger.info(f"Loading current preds: {current_path}")
    cur = read_preds(spark, current_path)

    logger.info(f"Loading baseline preds: {baseline_path}")
    base = read_preds(spark, baseline_path)

    # summaries
    cur_summary = summarize(cur["prediction_proba"].values)
    base_summary = summarize(base["prediction_proba"].values)

    # stat tests
    try:
        ks_stat, ks_p = ks_2samp(base["prediction_proba"].values, cur["prediction_proba"].values)
    except Exception:
        ks_stat, ks_p = None, None

    psi_val = psi(base["prediction_proba"].values, cur["prediction_proba"].values, bins=10)

    # decile diffs table
    base_dec = np.quantile(base["prediction_proba"].values, np.linspace(0, 1, 11))
    cur_dec = np.quantile(cur["prediction_proba"].values, np.linspace(0, 1, 11))
    decile_table = [{"decile": i, "base": float(base_dec[i]), "current": float(cur_dec[i]),
                     "diff": float(cur_dec[i] - base_dec[i])} for i in range(len(base_dec))]

    report = {
        "model_label": args.model_label,
        "snapshotdate": args.snapshotdate,
        "baseline_date": args.baseline_date,
        "timestamp": datetime.utcnow().isoformat(),
        "baseline_summary": base_summary,
        "current_summary": cur_summary,
        "ks_stat": None if ks_stat is None else float(ks_stat),
        "ks_pvalue": None if ks_p is None else float(ks_p),
        "psi": psi_val,
        "deciles": decile_table
    }

    # optional label metrics
    if args.labels_path:
        logger.info("Loading labels and computing supervised metrics")
        labels_sdf = spark.read.parquet(args.labels_path)
        labels_pdf = labels_sdf.toPandas()
        supervised = compute_metrics_with_labels(cur, labels_pdf, label_col=args.label_col)
        report["label_metrics"] = supervised

    # Save report
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"monitor_report_{args.snapshotdate.replace('-', '_')}.json"
    csv_path = out_dir / f"monitor_table_{args.snapshotdate.replace('-', '_')}.csv"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    # save decile table
    pd.DataFrame(decile_table).to_csv(csv_path, index=False)

    logger.info(f"Saved monitor report: {json_path}")
    logger.info(f"Saved decile table: {csv_path}")

    # MLflow logging (optional)
    if args.log_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri or "http://mlflow:5000")
        mlflow.set_experiment(args.mlflow_experiment or "model-monitoring")
        with mlflow.start_run(run_name=f"monitor_{args.model_label}_{args.snapshotdate}"):
            mlflow.log_metric("psi", float(psi_val) if psi_val is not None else math.nan)
            if ks_stat is not None:
                mlflow.log_metric("ks", float(ks_stat))
            # optional supervised logs
            if args.labels_path and report.get("label_metrics"):
                for k, v in report["label_metrics"].items():
                    if v is not None:
                        mlflow.log_metric(k, float(v))

            mlflow.log_artifact(str(json_path), artifact_path="monitor_report")
            mlflow.log_artifact(str(csv_path), artifact_path="monitor_report")

    # return nonzero if psi above critical threshold
    if args.psi_crit is not None and psi_val is not None:
        if psi_val >= args.psi_crit:
            logger.error(f"PSI {psi_val} >= critical {args.psi_crit}")
            exit(2)
        elif psi_val >= args.psi_warn:
            logger.warning(f"PSI {psi_val} >= warning {args.psi_warn}")
            exit(1)

    logger.info("Monitoring completed successfully.")
    exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_label", required=True, help="run id or model label used in prediction path")
    parser.add_argument("--snapshotdate", required=True, help="YYYY-MM-DD for current predictions")
    parser.add_argument("--baseline_date", required=True, help="YYYY-MM-DD for baseline predictions")
    parser.add_argument("--pred_path", default=None, help="explicit path to current prediction parquet (overrides model_label)")
    parser.add_argument("--baseline_path", default=None, help="explicit path to baseline prediction parquet")
    parser.add_argument("--output_dir", default="datamart/gold/model_monitoring/", help="where to write reports")
    parser.add_argument("--labels_path", default=None, help="optional parquet with true labels to compute supervised metrics")
    parser.add_argument("--label_col", default="label", help="label column name in labels parquet")
    parser.add_argument("--psi_warn", type=float, default=0.2, help="PSI warning threshold")
    parser.add_argument("--psi_crit", type=float, default=0.3, help="PSI critical threshold")
    parser.add_argument("--log_mlflow", action="store_true", help="log monitoring run to MLflow")
    parser.add_argument("--mlflow_tracking_uri", default=None)
    parser.add_argument("--mlflow_experiment", default=None)
    args = parser.parse_args()
    main(args)
