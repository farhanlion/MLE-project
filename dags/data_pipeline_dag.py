from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
with DAG(
    'data_pipeline_dag',
    default_args=default_args,
    description='Daily data pipeline for bronze, silver, and gold processing',
    schedule_interval='0 0 * * *', # run daily
    start_date=datetime(2016, 5, 1),
    end_date=datetime(2017, 3, 31),
    catchup=True,
) as dag:
    
    # data pipeline

    pipeline_start = DummyOperator(task_id="pipeline_start")

    # Bronze Processing
    bronze_table = BashOperator(
        task_id='bronze_table',
        bash_command=(
            'cd /app/airflow/scripts && '
            'python3 01_process_bronze.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # Silver Processing (run in parallel)
    silver_table_members = BashOperator(
        task_id='silver_table_members',
        bash_command=(
            'cd /app/airflow/scripts && '
            'python3 02_members_bronzetosilver.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_table_transactions = BashOperator(
        task_id='silver_table_transactions',
        bash_command=(
            'cd /app/airflow/scripts && '
            'python3 02_transactions_bronzetosilver.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_table_userlogs = BashOperator(
        task_id='silver_table_userlogs',
        bash_command=(
            'cd /app/airflow/scripts && '
            'python3 02_userlogs_bronzetosilver.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    silver_table_max_expiry_date = BashOperator(
        task_id='silver_table_max_expiry_date',
        bash_command=(
            'cd /app/airflow/scripts && '
            'python3 max_expiry_date.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # Gold Processing 

    gold_inference_feature_store = BashOperator(
        task_id="gold_inference_feature_store",        
        bash_command=(
            'cd /app/airflow/scripts && '
            'python3 inference_feature.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )

    # 1. Pipeline Start to Bronze
    pipeline_start >> bronze_table

    # 2. Bronze to ALL Silver tasks (run in parallel)
    # The list/tuple syntax means all tasks in the list/tuple will start once the preceding task (bronze_table) succeeds.
    bronze_table >> [
        silver_table_members,
        silver_table_transactions,
        silver_table_userlogs,
        silver_table_max_expiry_date
    ]

    # 3. ALL Silver tasks to Gold
    # The list/tuple syntax means the succeeding task (gold_inference_feature_store) will only start once ALL tasks in the list/tuple succeed.
    [
        silver_table_members,
        silver_table_transactions,
        silver_table_userlogs,
        silver_table_max_expiry_date
    ] >> gold_inference_feature_store