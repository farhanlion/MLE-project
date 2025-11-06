# Assumptions About your MLflow setup

## MLflow Tracking Server location

MLflow Tracking Server is reachable at:

`http://mlflow:5000`


(Adjust if needed.)

## Model name

You have registered models in MLflow under a registered name, e.g.: kkbox_churn_model

The script argument --modelname refers to:

MLflow registered model name, e.g.:

`--modelname "kkbox_churn_model"`

OR a specific stage: "Production", "Staging", etc.

OR a specific version: `"models:/kkbox_churn_model/3"`

## About preprocessing

Your training script logs the full model (WITH preprocessing) using:

mlflow.sklearn.log_model(best_model, "XGBoost_model")


Or similar for other algorithms.

Therefore, the MLflow model already contains preprocessing + model inside a Pipeline (this is the recommended way for MLflow).

So instead of manually loading:

model_artefact["preprocessing_transformers"]["stdscaler"]
model_artefact["model"]

→ You will call MLflow model directly:

model.predict_proba(X_inference)