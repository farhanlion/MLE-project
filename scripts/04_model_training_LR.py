#!/usr/bin/env python3
# coding: utf-8
"""
Train Logistic Regression only. Uses preprocessing_before_fit.prepare_data_for_training(...)
"""

import argparse
import logging
import os
import joblib

import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

from app.utils.model_preprocessor import prepare_data_for_training

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model, X, y, threshold=0.5):
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    metrics = {
        "roc_auc": roc_auc_score(y, proba),
        "precision": precision_score(y, preds, zero_division=0),
        "recall": recall_score(y, preds, zero_division=0),
        "f1": f1_score(y, preds, zero_division=0)
    }
    return metrics


def train_and_register(model_name, estimator, param_dist,
                       X_train, y_train, X_val, y_val, X_test, y_test, X_oot, y_oot,
                       n_iter, cv, random_state, scaler, lr_cols, categorical_cols,
                       mlflow_tracking_uri, mlflow_experiment):
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=f"{model_name}_tuning"):
        mlflow.set_tag("model_type", model_name)
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("random_state", random_state)

        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            return_train_score=True
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        logger.info(f"Best CV roc_auc: {search.best_score_:.4f}")
        mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
        for k, v in search.best_params_.items():
            mlflow.log_param(f"best_{k}", v)

        # evaluate
        for name, X, y in [("train", X_train, y_train), ("val", X_val, y_val),
                           ("test", X_test, y_test), ("oot", X_oot, y_oot)]:
            m = evaluate_model(best, X, y)
            for k, v in m.items():
                mlflow.log_metric(f"{name}_{k}", v)

        artifact_path = "LogisticRegression_model"
        mlflow.sklearn.log_model(best, artifact_path)

        # save preprocessing artifacts
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
        os.remove(scaler_path)

        lr_cols_path = "lr_feature_columns.txt"
        with open(lr_cols_path, "w") as f:
            f.write("\n".join(lr_cols))
        mlflow.log_artifact(lr_cols_path, artifact_path="preprocessing")
        os.remove(lr_cols_path)

        if categorical_cols:
            cc_path = "categorical_columns.txt"
            with open(cc_path, "w") as f:
                f.write("\n".join(categorical_cols))
            mlflow.log_artifact(cc_path, artifact_path="preprocessing")
            os.remove(cc_path)

        run_id = mlflow.active_run().info.run_id
        model_uri = f"runs:/{run_id}/{artifact_path}"
        try:
            reg_result = mlflow.register_model(model_uri=model_uri, name=model_name)
            logger.info(f"Registered model: {model_name} version={reg_result.version}")
            mlflow.set_tag("registered_model_name", model_name)
            mlflow.log_param("registered_model_version", reg_result.version)
        except Exception as e:
            logger.warning(f"Model registration failed: {e}")

        return best


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

    data = prepare_data_for_training(args)

    # model + param space
    lr = LogisticRegression(class_weight=data["class_weight_dict"], random_state=args.random_state, n_jobs=-1)
    lr_param_dist = {
        'C': uniform(0.01, 10),
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    }

    best_lr = train_and_register(
        "LogisticRegression",
        lr, lr_param_dist,
        data["X_train_lr"], data["y_train"],
        data["X_val_lr"], data["y_val"],
        data["X_test_lr"], data["y_test"],
        data["X_oot_lr"], data["y_oot"],
        n_iter=args.n_iter, cv=data["cv"], random_state=args.random_state,
        scaler=data["scaler"], lr_cols=data["lr_cols"], categorical_cols=data["categorical_cols"],
        mlflow_tracking_uri=args.mlflow_tracking_uri, mlflow_experiment=args.mlflow_experiment
    )

    logger.info("Logistic Regression training finished.")
