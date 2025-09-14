FROM apache/airflow:2.7.3

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy custom requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt