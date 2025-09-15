# CoverWallet Data Science Challenge

![Python](https://img.shields.io/badge/python-v3.12+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-orange.svg)
![Apache Airflow](https://img.shields.io/badge/Airflow-2.7.3-red.svg)

A complete ML solution for predicting account values using an XGBoost model, orchestrated with Apache Airflow and deployed via FastAPI, all containerized with Docker.

## 📋 Project Overview

This project tackles the CoverWallet data science challenge, which consists of predicting the account value of a given account using customer application data and initial quotes. The solution includes:

- **Data Processing Pipeline**: Feature engineering and preprocessing
- **Machine Learning Model**: XGBoost model achieving 95.06% R²
- **API Service**: FastAPI REST API for model serving
- **Orchestration**: Apache Airflow for workflow management
- **Containerization**: Docker and Docker Compose setup

## 🏗️ Project Structure

```
challenge_coverwallet/
├── api/
│   └── app/
│       ├── main.py              # FastAPI application
│       ├── requirements.txt     # API dependencies
│       └── Dockerfile          # API container setup
├── airflow/
│   ├── dags/
│   │   └── test_dag.py         # Airflow DAGs
│   ├── logs/                   # Airflow logs
│   ├── scripts/                # Airflow scripts
│   └── configs/                # Airflow configurations
├── data/
│   ├── accounts_train.csv      # Training accounts data
│   ├── quotes_train.csv        # Training quotes data
│   ├── accounts_test.csv       # Test accounts data
│   ├── quotes_test.csv         # Test quotes data
│   ├── features_train.csv      # Processed training features
│   ├── features_test.csv       # Processed test features
│   └── predictions.csv         # Model predictions
├── model/
│   └── xgboost_model.joblib    # Trained XGBoost model
├── notebooks/
│   ├── 1_eda.ipynb            # Exploratory Data Analysis
│   ├── 2_preprocessing.ipynb   # Data preprocessing
│   ├── 3_train.ipynb          # Model training
│   └── 4_analysis_results.ipynb # Results analysis
├── src/
│   ├── config/                 # Configuration files
│   ├── preprocessing_data.py   # Data preprocessing pipeline
│   ├── wrangling.py           # Data wrangling utilities
│   └── utils.py               # Utility functions
├── docker-compose.yml          # Multi-service orchestration
├── pyproject.toml             # Poetry dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- **Docker** and **Docker Compose** installed
- **Python 3.12+** (for local development)
- **Poetry** (for dependency management)

### Option 1: Docker Compose (Recommended)

This is the fastest way to get everything running:

```bash
# Clone the repository
git clone <repository-url>
cd challenge_coverwallet

# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build
```

**Services will be available at:**
- 🌐 **FastAPI**: http://localhost:8000
- 📊 **Airflow**: http://localhost:8080 (admin/admin)

### Option 2: Local Development with Poetry

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run data preprocessing (if needed)
poetry run python src/preprocessing_data.py --config src/config/config_preprocess.yaml --dataset test

# Start FastAPI locally
cd api/app
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 🐳 Docker Services

### FastAPI Service

The API service provides ML model predictions via REST endpoints.

**Build and run separately:**
```bash
# Build API container
docker build -f api/app/Dockerfile -t coverwallet-api .

# Run API container
docker run -p 8000:8000 -v $(pwd)/model:/model:ro coverwallet-api
```

**Environment Variables:**
- `MODEL_PATH`: Path to the XGBoost model file (default: `/model/xgboost_model.joblib`)

### Airflow Service

Orchestrates the ML pipeline and workflow management.

**Airflow Configuration:**
- **Executor**: SequentialExecutor
- **Database**: SQLite (development)
- **Default User**: admin/admin
- **Fernet Key**: Provided for encryption

**Access Airflow:**
1. Navigate to http://localhost:8080
2. Login with `admin`/`admin`
3. Explore DAGs and trigger workflows

## 🔧 API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Get Required Features
```bash
curl http://localhost:8000/feature-names
```

### Make Predictions
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

### Python Client Example
```python
import requests

# API endpoint
API_URL = "http://localhost:8000"

# Prepare features (all 37 required)
features = {
    "log_total_payroll": 15.2,
    "year_established": 2010.0,
    # ... include all 37 features
}

# Make prediction
response = requests.post(
    f"{API_URL}/predict",
    json={"features": features}
)

if response.status_code == 200:
    result = response.json()
    print(f"Predicted account value: ${result['prediction']:,.2f}")
else:
    print(f"Error: {response.text}")
```

## 📊 Model Information

### Performance Metrics
- **RMSE**: 663.80
- **MAE**: 161.48
- **R²**: 0.9506 (95.06% variance explained)

### Required Features (37 total)
The model requires exactly 37 features in a specific order:

```python
feature_names = [
    'log_total_payroll', 'year_established', 'total_payroll', 'product_concentration',
    'state_premium_sum_encoded', 'subindustry_sum_premium_encoded', 'carrier_concentration',
    'business_structure_revenue_encoded', 'premium_per_employee', 'industry_revenue_encoded',
    'num_quotes', 'revenue_x_payroll', 'log_annual_revenue', 'premium_to_revenue_ratio',
    'total_quotes', 'premium_ratio_max_avg', 'annual_revenue', 'max_x_nquotes',
    'state_revenue_encoded', 'num_employees', 'business_structure_premium_sum_encoded',
    'industry_sum_premium_encoded', 'num_products_requested', 'iqr_premium',
    'premium_per_revenue', 'avg_x_nproducts', 'subindustry_revenue_encoded',
    'premium_per_quote', 'max_premium', 'quotes_per_million_revenue', 'avg_premium',
    'sum_premium', 'carrier_diversity', 'quotes_per_employee', 'min_premium',
    'num_carriers', 'revenue_per_employee'
]
```

### Target Variable
- **Variable**: `account_value` (sum of premiums for converted products)
- **Range**: $29.25 - $134,752.41
- **Mean**: $1,723.70
- **Distribution**: Right-skewed with some high-value outliers

## 🔄 ML Pipeline

### Complete Pipeline with Airflow

The project includes a complete ML pipeline orchestrated with Apache Airflow:

```bash
# Start all services
docker-compose up -d --build

# Access Airflow UI
open http://localhost:8080  # admin/admin

# Execute the pipeline:
# 1. Find 'coverwallet_ml_pipeline' DAG
# 2. Toggle it ON
# 3. Click "Trigger DAG"
```

**Pipeline Steps:**
1. **Preprocessing**: Process accounts and quotes data
2. **Feature Engineering**: Create ML features 
3. **Prediction**: Generate account_value predictions
4. **Validation**: Validate results format and content

### Local Pipeline Execution

```bash
# Make script executable
chmod +x test_pipeline.sh

# Run complete pipeline
./test_pipeline.sh

# Or step by step:
poetry run python src/preprocessing_data.py --config src/config/config_preprocess.yaml --dataset test
poetry run python src/wrangling.py --config src/config/config_wrangling.yaml --dataset test
poetry run python src/detect.py --model-path model/xgboost_model.joblib --features-file features_test.csv --output-file predictions.csv
```

**Generated Files:**
- `data/accounts_test_processed.csv` - Processed accounts data
- `data/features_test.csv` - ML features for prediction
- `data/predictions.csv` - Final predictions (account_uuid, account_value)



### Running Notebooks
```bash
# Start Jupyter
poetry run jupyter lab

# Navigate to notebooks/ directory
# Available notebooks:
# - 1_eda.ipynb: Exploratory Data Analysis
# - 2_preprocessing.ipynb: Data preprocessing
# - 3_train.ipynb: Model training
# - 4_analysis_results.ipynb: Results analysis
```

### Data Processing Pipeline
```bash
# Run data wrangling
poetry run python src/wrangling.py --config src/config/config_wrangling.yaml --dataset test

# Run preprocessing
poetry run python src/preprocessing_data.py --config src/config/config_preprocess.yaml --dataset test
```

### Testing the API
```bash
# Health check
curl http://localhost:8000/health

# Interactive documentation
open http://localhost:8000/docs

# Alternative documentation
open http://localhost:8000/redoc
```

## 🔍 Monitoring and Logs

### Docker Logs
```bash
# View all services logs
docker-compose logs

# View specific service logs
docker-compose logs api
docker-compose logs airflow

# Follow logs in real-time
docker-compose logs -f api
```

### Airflow Monitoring
1. Access Airflow UI: http://localhost:8080
2. Monitor DAG runs and task status
3. View logs for individual tasks
4. Check system health and performance

## 🚨 Troubleshooting

### Common Issues

**Port already in use:**
```bash
# Kill processes using the ports
sudo lsof -ti:8000 | xargs kill -9
sudo lsof -ti:8080 | xargs kill -9

# Or use different ports
docker-compose up --build -p 8001:8000 -p 8081:8080
```

**Model file not found:**
```bash
# Ensure model file exists
ls -la model/xgboost_model.joblib

# Check volume mounts in docker-compose.yml
docker-compose config
```

**API returning 500 errors:**
```bash
# Check API logs
docker-compose logs api

# Verify all 37 features are provided
curl http://localhost:8000/feature-names
```

**Airflow not starting:**
```bash
# Reset Airflow database
docker-compose exec airflow airflow db reset

# Restart services
docker-compose restart airflow
```

### Resource Requirements
- **Minimum RAM**: 4GB
- **Recommended RAM**: 8GB
- **Disk Space**: 2GB for containers and data
- **CPU**: 2+ cores recommended

## 📈 Performance Tuning

### API Performance
- **Response Time**: < 100ms for predictions
- **Concurrency**: Supports multiple simultaneous requests
- **Memory Usage**: ~200MB per container

### Scaling Options
```bash
# Scale API service
docker-compose up --scale api=3

# Use production WSGI server
docker run -p 8000:8000 coverwallet-api gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📚 Additional Resources

### Documentation
- **FastAPI Docs**: http://localhost:8000/docs
- **Airflow Docs**: http://localhost:8080
- **Model Analysis**: See `findings.md` for detailed insights

### Key Files
- `findings.md`: Detailed model analysis and insights
- `pyproject.toml`: Poetry dependency management
- `docker-compose.yml`: Multi-service container orchestration
- `test_payload.json`: Sample payload for API testing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is part of the CoverWallet Data Science Challenge.

---

**Challenge**: CoverWallet Data Science Challenge  
**Last Updated**: September 2025