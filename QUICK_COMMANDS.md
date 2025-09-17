# CoverWallet Challenge - Quick Commands Reference

## 🚀 Start Everything
```bash
# Start all services (API + Airflow)
docker-compose up --build

# Start in background
docker-compose up -d --build

# Stop all services
docker-compose down
```

## 🌐 Access Services
- **FastAPI API**: http://localhost:8000
- **FastAPI Docs**: http://localhost:8000/docs  
- **Airflow UI**: http://localhost:8080 (admin/admin)

## 🧪 Quick Tests
```bash
# Health check
curl http://localhost:8000/health

# Get feature names
curl http://localhost:8000/feature-names

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @test_payload.json
```

## 📦 Poetry Commands
```bash
# Install dependencies
poetry install

# Activate environment
poetry shell

# Run preprocessing
poetry run python src/preprocessing_data.py --config src/config/config_preprocess.yaml --dataset test

# Start API locally
cd api/app && poetry run uvicorn main:app --reload
```

## 🔧 Troubleshooting
```bash
# Check logs
docker-compose logs api
docker-compose logs airflow

# Restart services
docker-compose restart

# Kill processes on ports
sudo lsof -ti:8000 | xargs kill -9
sudo lsof -ti:8080 | xargs kill -9
```

