from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from airflow.utils.task_group import TaskGroup

DUMMY_DATE = "2016-04-01"

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="scheduled_training_dag",
    description="One-shot training using today's date as a dummy cutoff.",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),  # any past date is fine
    schedule_interval="@once",        # run once (trigger when ready)
    catchup=False,
) as dag:

    start = EmptyOperator(task_id="start")

    with TaskGroup(group_id="train_ML_models") as training:
        trainxgb = BashOperator(
            task_id="run_XGB",
            bash_command=(
                f"python /app/scripts/04_model_training_XGB.py "
                f"--train_date {DUMMY_DATE}"
            )
        )
        trainlr = BashOperator(
            task_id="run_LR",
            bash_command=(
                f"python /app/scripts/04_model_training_LR.py "
                f"--train_date {DUMMY_DATE}"
            )
        )
        trainrf = BashOperator(
            task_id="run_RF",
            bash_command=(
                f"python /app/scripts/04_model_training_RF.py "
                f"--train_date {DUMMY_DATE}"
            )
        )

    end = EmptyOperator(task_id="end")

    start >> training >> end
