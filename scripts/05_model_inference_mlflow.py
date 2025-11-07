# robust_inference.py — drop-in snippet
import os
import joblib
import pandas as pd
import pyspark
from pyspark.sql.functions import col

# CONFIG
SNAPSHOT = "2016-04-10"
FEATURE_BASE = "/app/datamart/gold/feature_store/"
PARTITION = os.path.join(FEATURE_BASE, f"snapshot_date={SNAPSHOT}")
MODEL_PKL = "/app/mlflow/models/lr_churn_model_latest.pkl"

# Start Spark (quiet)
spark = pyspark.sql.SparkSession.builder.master("local[*]").appName("inference_safe").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load partition (fallback to base+filter)
if os.path.exists(PARTITION):
    sdf = spark.read.parquet(PARTITION)
else:
    if not os.path.exists(FEATURE_BASE):
        raise FileNotFoundError(FEATURE_BASE)
    sdf = spark.read.parquet(FEATURE_BASE)
    if "snapshot_date" in sdf.columns:
        sdf = sdf.filter(col("snapshot_date") == SNAPSHOT)

if sdf.count() == 0:
    raise ValueError("No rows for snapshot")

pdf = sdf.toPandas()

# Ensure id/snapshot_date exist
if "snapshot_date" not in pdf.columns:
    pdf["snapshot_date"] = SNAPSHOT
# rename common id synonyms to msno
for c in list(pdf.columns):
    if c.lower() in ("msno", "member_id", "user_id", "userid", "id", "memberid") and c != "msno":
        pdf = pdf.rename(columns={c: "msno"})
        break
if "msno" not in pdf.columns:
    raise KeyError("msno not found")

# Load model artifact (dict)
artifact = joblib.load(MODEL_PKL)
model = artifact.get("model")
scaler = artifact.get("scaler", None)
feature_cols = artifact.get("feature_columns")           # final OHE columns expected by model
numeric_cols_meta = artifact.get("numeric_columns", [])  # numeric columns list saved at training
orig_feature_list = artifact.get("original_feature_columns") or artifact.get("original_feature_list")


# --- Build X aligned to feature_cols ---
if feature_cols:
    # Determine raw categorical columns to OHE:
    if orig_feature_list:
        # choose candidates that actually exist in pdf
        candidates = [c for c in orig_feature_list if c in pdf.columns]
    else:
        candidates = [c for c in ("registered_via", "city_clean") if c in pdf.columns]

    # pick those candidate columns that are non-numeric (categorical)
    cat_cols = [c for c in candidates if not pd.api.types.is_numeric_dtype(pdf[c])]

    # Run one-hot (drop_first=True to match training)
    if cat_cols:
        pdf_ohe = pd.get_dummies(pdf, columns=cat_cols, drop_first=True, dtype=int)
    else:
        pdf_ohe = pdf.copy()

    # Add any missing expected model cols with zeros
    for c in feature_cols:
        if c not in pdf_ohe.columns:
            pdf_ohe[c] = 0

    # Keep only model columns (in saved order)
    X = pdf_ohe[feature_cols].copy()

else:
    # No final feature list — fallback to numeric-only selection excluding id/date
    drop = {"msno", "snapshot_date"}
    X = pdf[[c for c in pdf.columns if c not in drop and pd.api.types.is_numeric_dtype(pdf[c])]].copy()
    if X.shape[1] == 0:
        raise RuntimeError("No usable features and no feature_columns in pickle")

# --- Convert to numeric safely and fill NaNs ---
# coerce every column to numeric (non-numeric -> NaN), then fill with 0
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

# --- Apply scaler safely ---
if scaler is not None:
    try:
        # prefer scaler.feature_names_in_ if available (ensures correct order)
        if hasattr(scaler, "feature_names_in_"):
            scaler_cols = [c for c in scaler.feature_names_in_ if c in X.columns]
            if scaler_cols:
                X.loc[:, scaler_cols] = scaler.transform(X[scaler_cols])
        elif numeric_cols_meta:
            scaler_cols = [c for c in numeric_cols_meta if c in X.columns]
            if scaler_cols:
                X.loc[:, scaler_cols] = scaler.transform(X[scaler_cols])
        else:
            # fallback: scale all numeric columns
            num_cols = X.select_dtypes(include=["number"]).columns.tolist()
            if num_cols:
                X.loc[:, num_cols] = scaler.transform(X[num_cols])
    except Exception as e:
        # scaling failed — safe fallback: continue with filled but unscaled features
        print("Warning: scaler.transform failed:", str(e))

# Final safety net: ensure no NaN remains
if X.isnull().values.any():
    X = X.fillna(0.0)

# --- Predict probabilities ---
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X)[:, 1]
elif hasattr(model, "predict"):
    print("predict_proba not available; using predict as fallback")
    probs = pd.Series(model.predict(X)).astype(float).values
else:
    raise RuntimeError("Model has no predict/probability methods")




# Attach and show
pdf["churn_proba"] = probs
print(pdf[["msno", "snapshot_date", "churn_proba"]].to_string(index=False))


# --- Add metadata column for model file ---
pdf["model_file"] = os.path.basename(MODEL_PKL)

# --- Save predictions to parquet ---
OUT_BASE = "/app/datamart/gold/predictions"
os.makedirs(OUT_BASE, exist_ok=True)
out_path = os.path.join(OUT_BASE, f"predictions_{SNAPSHOT.replace('-', '_')}.parquet")

# Columns to persist
save_df = pdf[["msno", "snapshot_date", "churn_proba", "model_file"]].copy()

# Write to parquet (overwrite)
spark.createDataFrame(save_df).write.mode("overwrite").parquet(out_path)

print(f"Saved predictions to: {out_path}")


spark.stop()
