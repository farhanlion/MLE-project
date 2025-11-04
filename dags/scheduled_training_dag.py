from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowSkipException
from datetime import datetime, timedelta

# Wait for data_pipeline to complete
UPSTREAM_DAG_ID = 'data_pipeline_dag'
UPSTREAM_TASK_ID = 'gold_inference_feature_store'

# Path to store training metadata
METADATA_PATH = '/app/airflow/training_metadata.json'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 0, # Training should usually not auto-retry, handle failures manually
    'max_active_runs': 1, # Only allow one training run at a time
}

def get_last_training_date():
    """Get the last training date from metadata file"""
    try:
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
                return datetime.fromisoformat(metadata['last_training_date'])
    except Exception:
        pass
    return None

def update_last_training_date():
    """Update the last training date to current time"""
    trigger_source = context['dag_run'].conf.get('reason', 'scheduled')
    metadata = {
        'last_training_date': datetime.now().isoformat(),
        'updated_by': 'scheduled_training_dag',
        'trigger_source': trigger_source  # Track what triggered this training
    }
    os.makedirs(os.path.dirname(METADATA_PATH), exist_ok=True)
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)

def check_training_condition(**context):
    """
    Check if scheduled training should run:
    - First training ever
    - OR 6 months have passed since last training (scheduled or adhoc)
    """
    last_training = get_last_training_date()
    execution_date = context['execution_date']
    
    if last_training is None:
        print("✅ First training run - proceeding")
        return True
    
    six_months_after_last = last_training + timedelta(days=180)  # ~6 months
    should_run = execution_date >= six_months_after_last
    
    if should_run:
        print(f"✅ 6 months have passed since last training on {last_training.date()} - proceeding")
    else:
        print(f"⏸️  Only {(execution_date - last_training).days} days since last training on {last_training.date()} - skipping")
        raise AirflowSkipException("Not yet 6 months since last training")
    
    return should_run

# Define the DAG - now runs daily but only executes when condition met
with DAG(
    'scheduled_training_dag',
    default_args=default_args,
    description='Scheduled retraining pipeline that runs 6 months after last training (scheduled or adhoc).',
    schedule_interval='@daily',  # Check daily, but only run when condition met
    start_date=datetime(2016, 5, 1),
    catchup=False,
) as dag:
    
    pipeline_start = DummyOperator(task_id="pipeline_start")

    # 0. CONDITION CHECK: Only proceed if 6 months have passed since last training
    check_training_schedule = PythonOperator(
        task_id='check_training_schedule',
        python_callable=check_training_condition,
        provide_context=True
    )

    # 1. SENSOR: Wait for the features of the current run date to be ready
    wait_for_gold_features = ExternalTaskSensor(
        task_id='wait_for_gold_features_snapshot',
        external_dag_id=UPSTREAM_DAG_ID,
        external_task_id=UPSTREAM_TASK_ID,
        mode='reschedule',
        timeout=timedelta(hours=12).total_seconds(),
    )

    # 2. TRAINING TASK: Uses 12 months of data with latest month as OOT validation
    train_model = BashOperator(
        task_id='train_and_evaluate_model',
        bash_command=(
            'cd /app/airflow/scripts && '
            'python3 03_model_training.py '
            '--training_end_date "{{ ds }}" '           # Current execution date
            '--lookback_months 13 '                     # Get 13 months total
            '--train_months 12 '                        # Use first 12 months for training
            '--oot_months 1 '                           # Use latest month for OOT validation
            '--output_model_path /models/candidate '
            '--trigger_source "{{ dag_run.conf.get("reason", "scheduled") }}" ' 
        ),
    )

    # 3. EVALUATION & REGISTRATION: Check new model against production
    evaluate_and_promote = BashOperator(
        task_id='evaluate_and_promote_model',
        bash_command=(
            'cd /app/airflow/scripts && '
            'python3 06_evaluate_and_register.py '
            '--model_candidate /models/candidate '
            '--metrics_path /mlflow/training_run_metrics.json'
            '--oot_validation_period 1 '                # Specify OOT period for evaluation
        ),
    )

    # 4. UPDATE METADATA: Record this training run
    update_training_metadata = PythonOperator(
        task_id='update_training_metadata',
        python_callable=update_last_training_date
    )

    # 5. Cleanup Task
    training_pipeline_complete = DummyOperator(
        task_id='training_pipeline_complete'
    )

    # --- Set Dependencies ---
    (
        pipeline_start
        >> check_training_schedule
        >> wait_for_gold_features
        >> train_model
        >> evaluate_and_promote
        >> update_training_metadata
        >> training_pipeline_complete
    )