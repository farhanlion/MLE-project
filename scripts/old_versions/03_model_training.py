#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from scipy.stats import uniform, randint

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import argparse

import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# --------------------------------------------------------------------------------------
# MLflow Setup
# --------------------------------------------------------------------------------------
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("kkbox-churn-prediction")


# --------------------------------------------------------------------------------------
# Argument Parser
# --------------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="KKBox Churn Model Training Pipeline")

parser.add_argument(
    "--data_path",
    type=str,
    default="data_pdf.csv",
    help="Path to the input dataset CSV file."
)

parser.add_argument(
    "--train_date",
    type=str,
    default="2016-04-01",
    help="Model training cutoff date in YYYY-MM-DD format."
)

args = parser.parse_args()


# --------------------------------------------------------------------------------------
# Date Configuration
# --------------------------------------------------------------------------------------
model_train_date = datetime.strptime(args.train_date, "%Y-%m-%d")
train_months, val_months, test_months, oot_months = 8, 2, 2, 2

config = {}
config["oot_end"] = model_train_date - timedelta(days=1)
config["oot_start"] = model_train_date - relativedelta(months=oot_months)
config["test_end"] = config["oot_start"] - timedelta(days=1)
config["test_start"] = config["oot_start"] - relativedelta(months=test_months)
config["val_end"] = config["test_start"] - timedelta(days=1)
config["val_start"] = config["test_start"] - relativedelta(months=val_months)
config["train_end"] = config["val_start"] - timedelta(days=1)
config["train_start"] = config["val_start"] - relativedelta(months=train_months)
config["data_start_date"] = config["train_start"]
config["data_end_date"]   = config["oot_end"]



# --------------------------------------------------------------------------------------
# Load Data
# --------------------------------------------------------------------------------------
# data = pd.read_csv(args.data_path, parse_dates=["snapshot_date"])
# data["snapshot_date"] = data["snapshot_date"].dt.date

# ------------------------------------------------------
# LOAD SPARK PARQUET
# ------------------------------------------------------

features_path = "/app/datamart/gold/training_feature_store/"
labels_path   = "/app/datamart/gold/training_label_store/"

features_sdf = (
    spark.read.parquet(features_path)
         .filter(
            (col("snapshot_date") >= F.lit(config["data_start_date"])) &
            (col("snapshot_date") <= F.lit(config["data_end_date"]))
         )
)

labels_sdf = (
    spark.read.parquet(labels_path)
         .filter(
            (col("snapshot_date") >= F.lit(config["data_start_date"])) &
            (col("snapshot_date") <= F.lit(config["data_end_date"]))
         )
)


def slice_data(df, start, end):
    return df[
        (df["snapshot_date"] >= start.date()) &
        (df["snapshot_date"] <= end.date())
    ]

train = slice_data(data, config["train_start"], config["train_end"])
val   = slice_data(data, config["val_start"],   config["val_end"])
test  = slice_data(data, config["test_start"],  config["test_end"])
oot   = slice_data(data, config["oot_start"],   config["oot_end"])

feature_cols = [
    'tenure_days_at_snapshot', 'registered_via', 'city_clean', 
    'sum_secs_w30', 'active_days_w30', 'complete_rate_w30',
    'sum_secs_w7', 'engagement_ratio_7_30', 'days_since_last_play',
    'trend_secs_w30', 'auto_renew_share', 'last_is_auto_renew'
]

X_train, y_train = train[feature_cols], train["label"]
X_val,   y_val   = val[feature_cols],   val["label"]
X_test,  y_test  = test[feature_cols],  test["label"]
X_oot,   y_oot   = oot[feature_cols],   oot["label"]


# --------------------------------------------------------------------------------------
# Feature Engineering (Missing Flags + Fill)
# --------------------------------------------------------------------------------------
for df in [X_train, X_val, X_test, X_oot]:
    df["is_missing_activity"] = df["sum_secs_w30"].isnull().astype(int)
    df["is_missing_demo"] = df["tenure_days_at_snapshot"].isnull().astype(int)
    df.fillna(0, inplace=True)


# --------------------------------------------------------------------------------------
# Logistic Regression Preprocessing (One-Hot + Scaling)
# --------------------------------------------------------------------------------------
cat_cols = ["registered_via", "city_clean"]

def prepare_lr(df, train_cols=None, scaler=None, fit=False):
    df_ohe = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)

    if train_cols is None:
        train_cols = df_ohe.columns

    # Align columns
    for col in train_cols:
        if col not in df_ohe:
            df_ohe[col] = 0
    df_ohe = df_ohe[train_cols]

    # Identify numeric columns for scaling
    numeric_cols = [
        col for col in df_ohe.columns
        if col not in df_ohe.columns[df_ohe.dtypes == "uint8"] and
           col not in ["is_missing_activity", "is_missing_demo", "last_is_auto_renew"]
    ]

    if fit:
        scaler.fit(df_ohe[numeric_cols])

    df_ohe[numeric_cols] = scaler.transform(df_ohe[numeric_cols])
    return df_ohe, train_cols, scaler


scaler = StandardScaler()
X_train_lr, lr_cols, scaler = prepare_lr(X_train, fit=True, scaler=scaler)
X_val_lr, _, _ = prepare_lr(X_val, train_cols=lr_cols, scaler=scaler)
X_test_lr, _, _ = prepare_lr(X_test, train_cols=lr_cols, scaler=scaler)
X_oot_lr, _, _ = prepare_lr(X_oot, train_cols=lr_cols, scaler=scaler)


# --------------------------------------------------------------------------------------
# Tree-Based Model Data (Label Encoded)
# --------------------------------------------------------------------------------------
X_train_tree = X_train.copy()
X_val_tree   = X_val.copy()
X_test_tree  = X_test.copy()
X_oot_tree   = X_oot.copy()


# --------------------------------------------------------------------------------------
# Model Evaluation
# --------------------------------------------------------------------------------------
def evaluate(model, X, y, label):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return {
        f"{label}_roc_auc": roc_auc_score(y, proba),
        f"{label}_precision": precision_score(y, preds, zero_division=0),
        f"{label}_recall": recall_score(y, preds, zero_division=0),
        f"{label}_f1": f1_score(y, preds, zero_division=0)
    }


# --------------------------------------------------------------------------------------
# Training Function With MLflow Logging
# --------------------------------------------------------------------------------------
def train_model(
    model_name, estimator, param_dist, 
    X_train, y_train, 
    X_val, y_val, 
    X_test, y_test, 
    X_oot, y_oot,
    n_iter=10
):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    with mlflow.start_run(run_name=f"{model_name}_tuning"):

        mlflow.log_param("cv_folds", cv.n_splits)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_param("n_iter", n_iter)

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            return_train_score=True
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_
        mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
        mlflow.log_metric("best_cv_roc_auc", search.best_score_)

        # Evaluate across datasets
        metrics = {}
        for label, X, y in [
            ("train", X_train, y_train),
            ("val",   X_val,   y_val),
            ("test",  X_test,  y_test),
            ("oot",   X_oot,   y_oot)
        ]:
            m = evaluate(best_model, X, y, label)
            metrics.update(m)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(best_model, f"{model_name}_model")

        # --- Register best model to MLflow Model Registry ---
        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{model_name}_model"

        try:
            result = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            print(f"✅ Registered model: {model_name} (version={result.version})")

        except Exception as e:
            print(f"⚠️ Model registration failed: {e}")


        return best_model, metrics


# --------------------------------------------------------------------------------------
# Hyperparameter Search Spaces
# --------------------------------------------------------------------------------------
lr_params = {
    "C": uniform(0.01, 10),
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"],
    "max_iter": [1000, 2000, 3000]
}

rf_params = {
    "n_estimators": randint(100, 500),
    "max_depth": [None] + list(range(5, 30, 5)),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False]
}

xgb_params = {
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.3),
    "n_estimators": randint(100, 500),
    "min_child_weight": randint(1, 10),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "gamma": uniform(0, 5),
    "reg_alpha": uniform(0, 1),
    "reg_lambda": uniform(0, 1)
}

# --------------------------------------------------------------------------------------
# Train Logistic Regression
# --------------------------------------------------------------------------------------
class_weights = compute_class_weight("balanced", classes=[0, 1], y=y_train)
cw = {0: class_weights[0], 1: class_weights[1]}

lr_base = LogisticRegression(class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1)

lr_model, lr_metrics = train_model(
    "LogisticRegression", lr_base, lr_params,
    X_train_lr, y_train, 
    X_val_lr,   y_val,
    X_test_lr,  y_test,
    X_oot_lr,   y_oot
)


# --------------------------------------------------------------------------------------
# Train XGBoost
# --------------------------------------------------------------------------------------
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

xgb_base = XGBClassifier(
    scale_pos_weight=scale_pos,
    random_state=RANDOM_STATE,
    eval_metric="auc",
    use_label_encoder=False,
    n_jobs=-1
)

xgb_model, xgb_metrics = train_model(
    "XGBoost", xgb_base, xgb_params,
    X_train_tree, y_train,
    X_val_tree,   y_val,
    X_test_tree,  y_test,
    X_oot_tree,   y_oot
)


# --------------------------------------------------------------------------------------
# Train Random Forest
# --------------------------------------------------------------------------------------
rf_base = RandomForestClassifier(class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1)

rf_model, rf_metrics = train_model(
    "RandomForest", rf_base, rf_params,
    X_train_tree, y_train,
    X_val_tree,   y_val,
    X_test_tree,  y_test,
    X_oot_tree,   y_oot
)
