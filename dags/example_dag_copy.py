from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow'
}

# Define the DAG
with DAG(
    'dag123123',
    default_args=default_args,
    description='run immediately once',
    schedule_interval=None,
    start_date=datetime.now(),
    catchup=False,
) as dag:

    # Define the tasks
    task1 = BashOperator(
            task_id='check_pwd',
            bash_command='pwd'
    )

    task2 = BashOperator(
        task_id='first_hello',
        bash_command='python /opt/airflow/scripts/helloForAirflow.py'
    )


    task3 = BashOperator(
        task_id='second_hello',
        bash_command='python /opt/airflow/scripts/second_hello_to_airflow.py'
    )

    
    # Set the task order (task1 → task2)
    task1 >> task2 >> task3