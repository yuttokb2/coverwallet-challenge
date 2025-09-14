
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

def hello_world():
    print("Hello World from Airflow!")
    return "Hello World!"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'test_dag',
    default_args=default_args,
    description='A simple test DAG',
    schedule_interval=timedelta(days=1),
    catchup=False,
)

hello_task = PythonOperator(
    task_id='hello_world_task',
    python_callable=hello_world,
    dag=dag,
)