"""
DAG para el pipeline de ML de CoverWallet.
Ejecuta preprocessing, wrangling y predicción para datos de test.
"""

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

# Agregar el directorio src al path de Python
sys.path.insert(0, '/opt/airflow/src')

# Importar módulos del proyecto
from preprocessing_data import main as preprocessing_main
from wrangling import main as wrangling_main  
from detect import main as detect_main

# Configuración del logger
logger = logging.getLogger(__name__)

# Configuración por defecto del DAG
default_args = {
    'owner': 'coverwallet-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Configuración de paths
BASE_DIR = "/opt/airflow"
DATA_DIR = f"{BASE_DIR}/data"
SRC_DIR = f"{BASE_DIR}/src"
MODEL_DIR = f"{BASE_DIR}/model"

# Definir el DAG
dag = DAG(
    'coverwallet_ml_pipeline',
    default_args=default_args,
    description='Pipeline ML completo para CoverWallet: preprocessing, wrangling y predicción',
    schedule_interval=None,  # Ejecutar manualmente
    catchup=False,
    max_active_runs=1,
    tags=['ml', 'coverwallet', 'prediction']
)

def check_input_files(**context):
    """Verifica que los archivos de entrada existan."""
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
        raise FileNotFoundError(f"Archivos faltantes: {missing_files}")
    
    logger.info("Todos los archivos de entrada están disponibles")
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
        
        logger.info(f"Ejecutando preprocessing con argumentos: {sys.argv}")
        
        # CAMBIO: Verificar el código de retorno
        result = preprocessing_main()
        
        if result != 0:  # ← Aquí está el fix
            raise RuntimeError(f"preprocessing_main() falló con código: {result}")
        
        logger.info("Preprocessing completado exitosamente")
        
        # Verificar archivo de salida
        output_file = f"{DATA_DIR}/accounts_test_processed.csv"
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Archivo de salida no encontrado: {output_file}")
        
        return "success"
        
    except Exception as e:
        logger.error(f"Error en preprocessing: {e}")
        raise  # ← Esto hará que el DAG falle correctamente
    finally:
        os.chdir(original_cwd)
        


def run_wrangling(**context):
    """Ejecuta el script de wrangling (feature engineering)."""
    try:
        # Cambiar directorio de trabajo
        original_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        
        # Configurar argumentos
        sys.argv = [
            'wrangling.py',
            '--config', f'{SRC_DIR}/config/config_wrangling.yaml',
            '--dataset', 'test'
        ]
        
        logger.info(f"Ejecutando wrangling con argumentos: {sys.argv}")
        
        # Ejecutar función main directamente
        result = wrangling_main()
        
        if result != 0:  # ← Fix aquí también
            raise RuntimeError(f"wrangling_main() falló con código: {result}")
        
        # Verificar que el archivo de salida se haya creado
        output_file = f"{DATA_DIR}/features_test.csv"
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Archivo de salida no encontrado: {output_file}")
        
        return "success"
        
    except Exception as e:
        logger.error(f"Error en wrangling: {e}")
        raise
    finally:
        # Restaurar directorio original
        os.chdir(original_cwd)

def run_prediction(**context):
    """Ejecuta el script de predicción."""
    try:
        # Cambiar directorio de trabajo
        original_cwd = os.getcwd()
        os.chdir(BASE_DIR)
        
        # Configurar argumentos
        sys.argv = [
            'detect.py',
            '--model-path', f'{MODEL_DIR}/xgboost_model.joblib',
            '--features-file', 'features_test.csv',
            '--output-file', 'predictions.csv',
            '--base-dir', BASE_DIR
        ]
        
        logger.info(f"Ejecutando predicción con argumentos: {sys.argv}")
        
        # Ejecutar función main directamente
        detect_main()
        
        logger.info("Predicción completada exitosamente")
        
        # Verificar que el archivo de salida se haya creado
        output_file = f"{DATA_DIR}/predictions.csv"
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"Archivo de salida no encontrado: {output_file}")
        
        return "success"
        
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        raise
    finally:
        # Restaurar directorio original
        os.chdir(original_cwd)

def validate_results(**context):
    """Valida los resultados finales."""
    try:
        predictions_file = f"{DATA_DIR}/predictions.csv"
        
        # Cargar y validar predicciones
        df = pd.read_csv(predictions_file)
        
        # Verificaciones básicas
        required_columns = ['account_uuid', 'account_value']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Columnas faltantes en predictions.csv: {missing_columns}")
        
        # Verificar que no hay valores nulos
        if df.isnull().any().any():
            raise ValueError("Encontrados valores nulos en las predicciones")
        
        # Verificar que account_value es positivo
        if (df['account_value'] < 0).any():
            raise ValueError("Encontrados valores negativos en account_value")
        
        # Estadísticas básicas
        stats = {
            'total_predictions': len(df),
            'min_value': df['account_value'].min(),
            'max_value': df['account_value'].max(),
            'mean_value': df['account_value'].mean(),
            'median_value': df['account_value'].median()
        }
        
        logger.info("Validación completada exitosamente")
        logger.info(f"Estadísticas de predicciones: {stats}")
        
        # Guardar estadísticas en contexto para siguientes tareas
        context['task_instance'].xcom_push(key='prediction_stats', value=stats)
        
        return "success"
        
    except Exception as e:
        logger.error(f"Error en validación: {e}")
        raise

def send_notification(**context):
    """Envía notificación de finalización del pipeline."""
    try:
        # Obtener estadísticas de la tarea anterior
        stats = context['task_instance'].xcom_pull(
            task_ids='validate_results', 
            key='prediction_stats'
        )
        
        message = f"""
        Pipeline ML de CoverWallet completado exitosamente!
        
        Estadísticas:
        - Total de predicciones: {stats['total_predictions']}
        - Valor mínimo: ${stats['min_value']:,.2f}
        - Valor máximo: ${stats['max_value']:,.2f}
        - Valor promedio: ${stats['mean_value']:,.2f}
        - Valor mediano: ${stats['median_value']:,.2f}
        
        Archivo generado: data/predictions.csv
        """
        
        logger.info(message)
        return "success"
        
    except Exception as e:
        logger.error(f"Error enviando notificación: {e}")
        # No fallar el DAG por problemas de notificación
        return "notification_failed"

# Definir tareas del DAG

# Tarea inicial
start_task = DummyOperator(
    task_id='start_pipeline',
    dag=dag
)

# Verificar archivos de entrada
check_files_task = PythonOperator(
    task_id='check_input_files',
    python_callable=check_input_files,
    dag=dag
)

# Preprocessing
preprocessing_task = PythonOperator(
    task_id='run_preprocessing',
    python_callable=run_preprocessing,
    dag=dag
)

# Wrangling/Feature Engineering
wrangling_task = PythonOperator(
    task_id='run_wrangling',
    python_callable=run_wrangling,
    dag=dag
)

# Predicción
prediction_task = PythonOperator(
    task_id='run_prediction',
    python_callable=run_prediction,
    dag=dag
)

# Validación de resultados
validation_task = PythonOperator(
    task_id='validate_results',
    python_callable=validate_results,
    dag=dag
)

# Notificación final
notification_task = PythonOperator(
    task_id='send_notification',
    python_callable=send_notification,
    dag=dag
)

# Tarea final
end_task = DummyOperator(
    task_id='end_pipeline',
    dag=dag
)

# Definir dependencias del DAG
start_task >> check_files_task >> preprocessing_task >> wrangling_task >> prediction_task >> validation_task >> notification_task >> end_task

# Configuración adicional del DAG
dag.doc_md = """
# CoverWallet ML Pipeline

Este DAG ejecuta el pipeline completo de machine learning para generar predicciones de account_value:

## Tareas:

1. **check_input_files**: Verifica que existan los archivos necesarios
2. **run_preprocessing**: Ejecuta preprocessing de datos
3. **run_wrangling**: Ejecuta feature engineering 
4. **run_prediction**: Genera predicciones con el modelo XGBoost
5. **validate_results**: Valida los resultados generados
6. **send_notification**: Envía notificación de finalización

## Archivos de entrada:
- `data/accounts_test.csv`
- `data/quotes_test.csv` 
- `model/xgboost_model.joblib`

## Archivos de salida:
- `data/accounts_test_processed.csv`
- `data/features_test.csv`
- `data/predictions.csv`

## Configuración:
- Ejecutar manualmente (no programado)
- Reintentos: 1
- Timeout de reintento: 5 minutos
"""
