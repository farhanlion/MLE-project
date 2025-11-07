from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
}

with DAG(
    'data_pipeline_dag',
    default_args=default_args,
    description='Manual data pipeline for bronze, silver, and gold processing',
    schedule_interval=None,   # 👈 disables automatic scheduling
    start_date=datetime(2025, 1, 1),  # safe placeholder
    catchup=False,             # 👈 ensures no backfill runs
) as dag:
    
    # data pipeline

    pipeline_start = DummyOperator(task_id="pipeline_start")

    # Bronze Processing
    bronze_members = BashOperator(
    task_id='bronze_members_table',
    bash_command="""
            python /app/scripts/01_bronze_members.py
        """,
    )

    bronze_transactions = BashOperator(
    task_id='bronze_transactions_table',
    bash_command="""
            python /app/scripts/01_bronze_transactions.py
        """,
    )

    bronze_userlogs = BashOperator(
    task_id='bronze_userlogs_table',
    bash_command="""
            python /app/scripts/01_bronze_userlogs.py
        """,
    )

    # Silver Processing (run in parallel)
    silver_max_expiry_date = BashOperator(
        task_id='silver_max_expiry_date',
        bash_command=(
            'python /app/scripts/02_silver_max_expirydate.py '
        ),
    )

    silver_members = BashOperator(
        task_id='silver_members',
        bash_command=(
            'python /app/scripts/02_silver_members.py '
        ),
    )

    silver_transactions = BashOperator(
        task_id='silver_transactions',
        bash_command=(
            'python /app/scripts/02_silver_transactions.py '
        ),
    )

    silver_latest_transactions = BashOperator(
        task_id='silver_latest_transactions',
        bash_command=(
            'python /app/scripts/02_silver_latest_transactions.py '
        ),
    )

    silver_userlogs = BashOperator(
        task_id='silver_userlogs',
        bash_command=(
            'python /app/scripts/02_silver_userlogs.py '
        ),
    )
    
    
    # Gold Processing 
    gold_feature_store = BashOperator(
        task_id="gold_feature_store",        
        bash_command=(
            'python /app/scripts/03_gold_feature_processing.py'
        ),
    )

    gold_label_store = BashOperator(
        task_id="gold_label_store",        
        bash_command=(
            'python /app/scripts/03_gold_label_processing.py'
        ),
    )

    pipeline_start >> [
    bronze_members,
    bronze_transactions,
    bronze_userlogs
    ] >> DummyOperator(task_id="silver_start") >> [
        silver_members,
        silver_transactions,
        silver_userlogs,
        silver_max_expiry_date,
        silver_latest_transactions
    ] >> DummyOperator(task_id="gold_start") >> [
        gold_feature_store,
        gold_label_store
    ]

