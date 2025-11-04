from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

# Wait for data_pipeline to complete
UPSTREAM_DAG_ID = 'data_pipeline_dag'
UPSTREAM_TASK_ID = 'gold_inference_feature_store'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0, # Training should usually not auto-retry, handle failures manually
    'max_active_runs': 1, # Only allow one training run at a time
}

# Define the DAG
with DAG(
    'scheduled_training_dag',
    default_args=default_args,
    description='Bi-annual scheduled retraining pipeline using features from data_pipeline_dag.',
    # Schedule: Run on the 1st day of the months specified (May and November)
    # Cron: 0 0 1 5,11 * (At 00:00, on day 1 of May and November)
    schedule_interval='0 0 1 5,11 *', 
    # Use a clear start date, allowing for the first run on the next scheduled date
    start_date=datetime(2017, 5, 1),
    catchup=False,
) as dag:
    
    # data pipeline

    pipeline_start = DummyOperator(task_id="pipeline_start")

# 1. SENSOR: Wait for the features of the current run date to be ready
    # We wait for the Upstream DAG to complete for the same date/interval.
    wait_for_gold_features = ExternalTaskSensor(
        task_id='wait_for_gold_features_snapshot',
        external_dag_id=UPSTREAM_DAG_ID,
        external_task_id=UPSTREAM_TASK_ID,
        mode='reschedule', # Recommended for long waits
        timeout=timedelta(hours=12).total_seconds(), # Give a generous window for data pipeline
    )

    # 2. TRAINING TASK: Pulls 6 months of data and trains the model
    # The training script will use the {{ ds }} as the end date for the training window
    train_model = BashOperator(
        task_id='train_and_evaluate_model',
        bash_command=(
            'cd /app/airflow/scripts && '
            'python3 05_train_model.py '
            '--training_end_date "{{ ds }}" '        # Use current run date as end of training window
            '--lookback_months 6 '                   # Pass parameters to define the training set
            '--output_model_path /models/candidate ' # Save as a temporary candidate model
        ),
    )

    # 3. EVALUATION & REGISTRATION: Check new model against production
    # This task compares the candidate model to the currently deployed model.
    # If the candidate is better, it promotes it (e.g., moves it to /models/production).
    evaluate_and_promote = BashOperator(
        task_id='evaluate_and_promote_model',
        bash_command=(
            'cd /app/airflow/scripts && '
            'python3 06_evaluate_and_register.py '
            '--model_candidate /models/candidate '
            '--metrics_path /mlflow/training_run_metrics.json'
        ),
    )

    # 4. Cleanup Task (Optional)
    training_pipeline_complete = EmptyOperator(
        task_id='training_pipeline_complete'
    )

    # --- Set Dependencies ---
    (
        wait_for_gold_features 
        >> train_model 
        >> evaluate_and_promote 
        >> training_pipeline_complete
    )