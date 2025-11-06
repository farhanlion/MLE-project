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
            'python /app/scripts/02_max_expiry_latest_txn_bronzetosilver.py '
        ),
    )

    silver_members = BashOperator(
        task_id='silver_members',
        bash_command=(
            'python /app/scripts/02_members_bronzetosilver.py '
        ),
    )

    silver_transactions = BashOperator(
        task_id='silver_transactions',
        bash_command=(
            'python /app/scripts/02_transactions_bronzetosilver.py '
        ),
    )

    silver_latest_transactions = BashOperator(
        task_id='silver_latest_transactions',
        bash_command=(
            'python /app/scripts/02_latest_transactions_bronzetosilver.py '
        ),
    )

    silver_userlogs = BashOperator(
        task_id='silver_userlogs',
        bash_command=(
            'python /app/scripts/02_userlogs_bronzetosilver.py '
        ),
    )
    
    
    # Gold Processing 
    gold_inference_feature_store = BashOperator(
        task_id="gold_feature_store",        
        bash_command=(
            'python /app/scripts/03_gold_feature_processing_v2.py --mode full'
        ),
    )

    gold_label_store = BashOperator(
        task_id="gold_label_store",        
        bash_command=(
            'python /app/scripts/03_gold_label_processing_v2.py --mode full'
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
        gold_inference_feature_store,
        gold_label_store
    ]

    





    # 3. ALL Silver tasks to Gold
    # The list/tuple syntax means the succeeding task (gold_inference_feature_store) will only start once ALL tasks in the list/tuple succeed.
    # [
    #     silver_table_members,
    #     silver_table_transactions,
    #     silver_table_userlogs,
    #     silver_table_max_expiry_date
    # ] >> gold_inference_feature_store