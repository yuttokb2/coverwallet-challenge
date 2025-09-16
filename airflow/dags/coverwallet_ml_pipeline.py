from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.utils.dates import days_ago
import pandas as pd
import sys
import os
import logging
from pathlib import Path

# Add the src dir to python's path
sys.path.insert(0, '/opt/airflow/src')


# Importing modules 
from preprocessing_data import main as preprocessing_main
from wrangling import main as wrangling_main  
from detect import main as detect_main


logger = logging.getLogger(__name__)


default_args = {
    'owner': 'coverwallet-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Setting paths
BASE_DIR = "/opt/airflow"
DATA_DIR = f"{BASE_DIR}/data"
SRC_DIR = f"{BASE_DIR}/src"
MODEL_DIR = f"{BASE_DIR}/model"


dag = DAG(
    'coverwallet_ml_pipeline',
    default_args=default_args,
    description='Full Pipeline ML for CoverWallet Challenge: preprocessing, wrangling and prediction',
    schedule_interval=None,  # Manual execution
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'coverwallet', 'prediction']
)

def check_input_files(**context):
    """Verify that the files exist"""
    required_files = [
        f"{DATA_DIR}/accounts_test.csv",
        f"{DATA_DIR}/quotes_test.csv",
        f"{MODEL_DIR}/xgboost_model.joblib"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}")
    
    logger.info("All files available!")
    return "success"

def run_preprocessing(**context):
    try:
        original_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        
        sys.argv = [
            'preprocessing_data.py',
            '--config', f'{SRC_DIR}/config/config_preprocess.yaml',
            '--dataset', 'test'
        ]
        
        logger.info(f"Running preprocessing.py: {sys.argv}")
        
        result = preprocessing_main()
        
        if result != 0:  # ← Aquí está el fix
            raise RuntimeError(f"preprocessing_main() failed with code: {result}")
        
        logger.info("Preprocessing completed successfully!")
        

        output_file = f"{DATA_DIR}/accounts_test_processed.csv"
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file not found: {output_file}")
        
        return "success"
        
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise 
    finally:
        os.chdir(original_cwd)
        


def run_wrangling(**context):
    try:
        original_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        
        # Configurar argumentos
        sys.argv = [
            'wrangling.py',
            '--config', f'{SRC_DIR}/config/config_wrangling.yaml',
            '--dataset', 'test'
        ]
        
        logger.info(f"Running wrangling.py: {sys.argv}")
        
        result = wrangling_main()
        
        if result != 0: 
            raise RuntimeError(f"wrangling_main() failed with code: {result}")

        output_file = f"{DATA_DIR}/features_test.csv"
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Output file not found: {output_file}")
        
        return "success"
        
    except Exception as e:
        logger.error(f"Error in wrangling: {e}")
        raise
    finally:
        os.chdir(original_cwd)

def run_prediction(**context):
    try:
        original_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        
        sys.argv = [
            'detect.py',
            '--model-path', f'{MODEL_DIR}/xgboost_model.joblib',
            '--features-file', 'features_test.csv',
            '--output-file', 'predictions.csv',
            '--base-dir', BASE_DIR
        ]
        
        logger.info(f"Running predict.py: {sys.argv}")
        
        # Ejecutar función main directamente
        detect_main()
        
        logger.info("Prediction completed!")

        output_file = f"{DATA_DIR}/predictions.csv"
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"File not found: {output_file}")
        
        return "success"
        
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        raise
    finally:

        os.chdir(original_cwd)

def validate_results(**context):
    try:
        predictions_file = f"{DATA_DIR}/predictions.csv"
        

        df = pd.read_csv(predictions_file)
        

        required_columns = ['account_uuid', 'account_value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing columns in predictions.csv: {missing_columns}")
        

        if df.isnull().any().any():
            raise ValueError("Found null values in predictions.csv")
        
        
        if (df['account_value'] < 0).any():
            raise ValueError("Found negative account values")
        
        stats = {
            'total_predictions': len(df),
            'min_value': df['account_value'].min(),
            'max_value': df['account_value'].max(),
            'mean_value': df['account_value'].mean(),
            'median_value': df['account_value'].median()
        }
        
        logger.info("Validation completed successfully.")
        logger.info(f"Prediction stats: {stats}")
    
        context['task_instance'].xcom_push(key='prediction_stats', value=stats)
        
        return "success"
        
    except Exception as e:
        logger.error(f"Error in validation: {e}")
        raise

def send_notification(**context):
    try:
        # Obtener estadísticas de la tarea anterior
        stats = context['task_instance'].xcom_pull(
            task_ids='validate_results', 
            key='prediction_stats'
        )
        
        message = f"""
        Pipeline ML in CoverWallet completed successfully!
        
        Stats:
        - Total predictions: {stats['total_predictions']}
        - Min value: ${stats['min_value']:,.2f}
        - Max value: ${stats['max_value']:,.2f}
        - Mean value: ${stats['mean_value']:,.2f}
        - Median value: ${stats['median_value']:,.2f}
        
        File generated at: data/predictions.csv
        """
        
        logger.info(message)
        return "success"
        
    except Exception as e:
        logger.error(f"Error while sending notification: {e}")
        return "notification_failed"


start_task = DummyOperator(
    task_id='start_pipeline',
    dag=dag
)


check_files_task = PythonOperator(
    task_id='check_input_files',
    python_callable=check_input_files,
    dag=dag
)


preprocessing_task = PythonOperator(
    task_id='run_preprocessing',
    python_callable=run_preprocessing,
    dag=dag
)


wrangling_task = PythonOperator(
    task_id='run_wrangling',
    python_callable=run_wrangling,
    dag=dag
)


prediction_task = PythonOperator(
    task_id='run_prediction',
    python_callable=run_prediction,
    dag=dag
)


validation_task = PythonOperator(
    task_id='validate_results',
    python_callable=validate_results,
    dag=dag
)


notification_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag
)


end_task = DummyOperator(
    task_id='end_pipeline',
    dag=dag
)

start_task >> check_files_task >> preprocessing_task >> wrangling_task >> prediction_task >> validation_task >> notification_task >> end_task

