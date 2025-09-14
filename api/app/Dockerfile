FROM python:3.12-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY ./api/app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code directamente al WORKDIR
COPY ./api/app .

# Create model directory and copy model
RUN mkdir -p /model
COPY ./model/xgboost_model.joblib /model/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application - ahora main.py está directamente en /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]