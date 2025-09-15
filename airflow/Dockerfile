FROM apache/airflow:2.9.3-python3.12

USER root
# Instalar dependencias del sistema si las necesitas
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

    
# Fix permisos para directorios de datos
RUN mkdir -p /opt/airflow/data /opt/airflow/model /opt/airflow/src/config && \
    chown -R airflow:root /opt/airflow/data /opt/airflow/model /opt/airflow/src && \
    chmod -R 775 /opt/airflow/data /opt/airflow/model /opt/airflow/src
USER airflow

# Copiar e instalar requirements
COPY airflow/requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Set Python path
ENV PYTHONPATH="/opt/airflow/src:${PYTHONPATH}"

