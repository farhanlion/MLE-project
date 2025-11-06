from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.dag import TriggerDagRunOperator
from airflow.utils.dates import days_ago
from airflow.models.dagrun import DagRun
from airflow.models.taskinstance import TaskInstance
from airflow.utils.state import State
from datetime import datetime, timedelta

# Daily monitoring - execute adhoc_training when metric is below threshold for 60 days
MONITORING_DAG_ID = 'daily_inference_dag'
MONITORING_TASK_ID = 'run_daily_monitoring'
TRAINING_DAG_ID = 'scheduled_training_dag'
FAILURE_THRESHOLD_DAYS = 60

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0, # Training should usually not auto-retry, handle failures manually
    'max_active_runs': 1, # Only allow one training run at a time
}

def check_sustained_monitoring_failure(ti=None):
    """
    Checks if the monitoring task failed for the last N consecutive days.
    """
    # Define the time range to check (e.g., last 60 days up to yesterday)
    end_date = ti.execution_date.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=FAILURE_THRESHOLD_DAYS - 1)

    # Get the DagRuns and TaskInstances for the monitoring task
    task_instances = TaskInstance.get_task_instances(
        dag_id=MONITORING_DAG_ID,
        task_ids=[MONITORING_TASK_ID],
        start_date=start_date,
        end_date=end_date,
    )

    if len(task_instances) < FAILURE_THRESHOLD_DAYS:
        # Not enough runs to meet the 60-day window yet
        print(f"Only {len(task_instances)} runs found, skipping check.")
        return False

    # Check if ALL instances in the window failed
    consecutive_failures = all(ti.state == State.FAILED for ti in task_instances)
    
    if consecutive_failures:
        print(f"Condition met: {FAILURE_THRESHOLD_DAYS} consecutive failures detected for {MONITORING_TASK_ID}.")
    else:
        # If any run succeeded, the condition is not met
        print("Condition NOT met: Monitoring task succeeded at least once in the window.")
    
    # Return the boolean result which will be used by the branching logic (via XComs)
    return consecutive_failures


# --- Define the DAG ---
with DAG(
    'metric_check_trigger_dag',
    start_date=days_ago(1),
    schedule_interval='@daily',
    catchup=False,
) as dag:
    
    check_metrics = PythonOperator(
        task_id='check_60_day_failure_window',
        python_callable=check_sustained_monitoring_failure,
        provide_context=True, # Allows access to `ti`
    )

    # Trigger the Training DAG if the Python task returned True
    trigger_training = TriggerDagRunOperator(
        task_id="trigger_adhoc_training",
        trigger_dag_id=TRAINING_DAG_ID,
        conf={"reason": "Sustained metric failure over 60 days"},
        # Only proceed if the result of check_metrics was True
        wait_for_completion=False,
        execution_date="{{ ds }}", # Pass the current execution date to the downstream DAG
        depends_on_past=False,
    )

    # Use a dummy/empty task as a safe path if training is NOT triggered
    no_trigger_needed = EmptyOperator(
        task_id="no_adhoc_training_needed",
    )

from airflow.operators.python import ShortCircuitOperator
    
    def decide_to_trigger(ti):
        result = ti.xcom_pull(task_ids='check_60_day_failure_window', key='return_value')
        return result
        
    branch_task = ShortCircuitOperator(
        task_id='decide_if_training_needed',
        python_callable=decide_to_trigger,
        depends_on_past=False,
    )
    
    check_metrics >> branch_task >> trigger_training
    check_metrics >> branch_task >> no_trigger_needed # This line is often omitted for ShortCircuit
