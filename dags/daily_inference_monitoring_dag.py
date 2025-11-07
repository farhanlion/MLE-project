from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook
from datetime import datetime, timedelta


# Slack failure alert
# def notify_slack(context):
#     dag_id = context["dag"].dag_id
#     task_id = context["task"].task_id
#     exec_dt = context["ts"]
#     log_url = context["task_instance"].log_url
#     msg = (
#         f":rotating_light: *Airflow Task Failed*\n"
#         f"• DAG: `{dag_id}`\n"
#         f"• Task: `{task_id}`\n"
#         f"• Time: `{exec_dt}`\n"
#         f"• <{log_url}|View Logs>"
#     )
#     SlackWebhookHook(slack_webhook_conn_id="slack_webhook").send(text=msg)


# ------------------------
# HARDCODED VALUES
# ------------------------
MODEL_FILE = "xgb_model_20251103_122821.pkl"
CURRENT_SNAPSHOT = "2016-04-01"
BASELINE_SNAPSHOT = "2016-05-01"

INFERENCE_SCRIPT = "/app/scripts/05_model_inference.py"
MONITORING_SCRIPT = "/app/scripts/06_monitor_predictions.py"

PRED_PATH_CUR = f"/app/datamart/gold/predictions/predictions_2016_04_01.parquet"
PRED_PATH_BASE = f"/app/datamart/gold/predictions/predictions_2016_05_01.parquet"
MONITOR_OUTPUT = f"/app/datamart/gold/model_monitoring/{MODEL_FILE}"
LABELS_PATH = "/app/datamart/gold/label_store"


default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    # "on_failure_callback": notify_slack,
}


with DAG(
    dag_id="batch_inference_and_monitoring",
    start_date=datetime(2025, 1, 1),
    schedule_interval="@once",
    catchup=False,
    default_args=default_args,
) as dag:

    start = EmptyOperator(task_id="start")

    # ------------------------
    # ✅ Inference task
    # ------------------------
    run_inference = BashOperator(
        task_id="run_inference",
        bash_command=f"""
python {INFERENCE_SCRIPT}
        """,
        execution_timeout=timedelta(minutes=30),
    )

    # ------------------------
    # ✅ Monitoring task
    # ------------------------
    run_monitor = BashOperator(
        task_id="run_prediction_monitor",
        bash_command="""
        python /app/scripts/06_monitor_predictions.py \
          --model_label xgb_model_20251103_122821.pkl \
          --snapshotdate 2016-04-01 \
          --baseline_date 2016-05-01 \
          --pred_path /app/datamart/gold/predictions/predictions_2016_04_01.parquet \
          --baseline_path /app/datamart/gold/predictions/predictions_2016_05_01.parquet \
          --output_dir /app/datamart/gold/model_monitoring/xgb_model_20251103_122821 \
          --labels_path /app/datamart/gold/label_store
        """
    )

    end = EmptyOperator(task_id="end")

    start >> run_inference >> run_monitor >> end
