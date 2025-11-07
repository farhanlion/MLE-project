import subprocess
import sys

print("\n🚀 RUNNING INFERENCE...\n")
inference_cmd = [
    "python", "/app/scripts/05_model_inference.py"
]

res1 = subprocess.run(inference_cmd)
if res1.returncode != 0:
    print("❌ Inference failed. Exiting pipeline.")
    sys.exit(res1.returncode)

print("\n✅ Inference completed.\n")
print("\n🚀 RUNNING MODEL MONITORING...\n")

monitor_cmd = [
    "python", "/app/scripts/06_monitor_predictions.py",
    "--model_label", "xgb_model_20251103_122821.pkl",
    "--snapshotdate", "2016-04-10",
    "--baseline_date", "2016-04-02",
    "--pred_path", "/app/datamart/gold/predictions/predictions_2016_05_01.parquet",
    "--baseline_path", "/app/datamart/gold/predictions/predictions_2016_04_10.parquet",
    "--output_dir", "/app/datamart/gold/model_monitoring/xgb_model_20251103_122821",
    "--labels_path", "/app/datamart/gold/label_store"
]

res2 = subprocess.run(monitor_cmd)
if res2.returncode != 0:
    print("❌ Monitoring failed.")
    sys.exit(res2.returncode)

print("\n✅ Monitoring completed. Pipeline finished successfully! 🎉")
