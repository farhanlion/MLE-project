from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime, timedelta
from airflow.providers.slack.hooks.slack_webhook import SlackWebhookHook

# --- Slack alert on failure ---
def notify_slack(context):
    dag_id = context["dag"].dag_id
    task_id = context["task"].task_id
    exec_dt = context["ts"]
    log_url = context["task_instance"].log_url
    msg = f":rotating_light: *Airflow Failure*\n• DAG: `{dag_id}`\n• Task: `{task_id}`\n• When: `{exec_dt}`\n• Logs: {log_url}"
    SlackWebhookHook(slack_webhook_conn_id="slack_webhook").send(text=msg)

DUMMY_DATE = "2016-01-01"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "on_failure_callback": notify_slack,
}

with DAG(
    dag_id="scheduled_inference_and_monitoring_dag",
    description="One-shot inference using a fixed dummy date (2016-01-01) with monitoring.",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval="@once",   # run once when triggered
    catchup=False,
) as dag:

    start = EmptyOperator(task_id="start")

    # --- Inference step ---
    inference = BashOperator(
        task_id="run_inference",
        bash_command=(
            f"python /app/scripts/05_model_inference_v2.py --inference_date {DUMMY_DATE}"
        ),
        execution_timeout=timedelta(minutes=30),
    )

    # --- Basic monitoring checks ---
    with TaskGroup(group_id="post_inference_checks") as checks:
        data_freshness = BashOperator(
            task_id="data_freshness_check",
            bash_command="python /app/monitoring/check_data_freshness.py --max_lag_hours 24",
            execution_timeout=timedelta(minutes=5),
        )

        output_quality = BashOperator(
            task_id="output_quality_check",
            bash_command="python /app/monitoring/check_output_quality.py --artifact /app/outputs/predictions.parquet",
            execution_timeout=timedelta(minutes=10),
        )

        drift_probe = BashOperator(
            task_id="drift_probe",
            bash_command="python /app/monitoring/quick_drift_check.py --ref /app/ref/feature_stats.json --cur /app/outputs/feature_stats.json --psi_warn 0.2 --psi_crit 0.3",
            execution_timeout=timedelta(minutes=10),
        )

    end = EmptyOperator(task_id="end")

    start >> inference >> checks >> end
