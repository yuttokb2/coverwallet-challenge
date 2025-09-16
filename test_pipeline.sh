#!/bin/bash
# Script para probar el pipeline completo de ML

set -e

echo "🚀 Starting Coverwallet ML pipeline..."

# Variables
BASE_DIR=$(pwd)
DATA_DIR="$BASE_DIR/data"
SRC_DIR="$BASE_DIR/src"
MODEL_DIR="$BASE_DIR/model"

echo "📁 Verifying required files..."


REQUIRED_FILES=(
    "$DATA_DIR/accounts_test.csv"
    "$DATA_DIR/quotes_test.csv" 
    "$MODEL_DIR/xgboost_model.joblib"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Archivo faltante: $file"
        exit 1
    else
        echo "✅ $file"
    fi
done

echo ""
echo "Running the pipeline step by step..."

# Paso 1: Preprocessing
echo "Running preprocessing..."
poetry run python src/preprocessing_data.py --config src/config/config_preprocess.yaml --dataset test
if [ $? -eq 0 ]; then
    echo "✅ Preprocessing completado"
else
    echo "❌ Error en preprocessing"
    exit 1
fi

# Paso 2: Wrangling/Feature Engineering
echo "Running wrangling (feature engineering)..."
poetry run python src/wrangling.py --config src/config/config_wrangling.yaml --dataset test
if [ $? -eq 0 ]; then
    echo "✅ Wrangling completado"
else
    echo "❌ Error en wrangling"
    exit 1
fi


echo "Running prediction..."
poetry run python src/detect.py --model-path model/xgboost_model.joblib --features-file features_test.csv --output-file predictions.csv
if [ $? -eq 0 ]; then
    echo "Prediction completed"
else
    echo "Error in prediction"
    exit 1
fi

# Verificar resultados
echo ""
echo "📊 Verifying results..."

if [ -f "$DATA_DIR/predictions.csv" ]; then
    echo "Predictions file saved in: $DATA_DIR/predictions.csv"
    

    echo ""
    echo "📈 Prediction stats:"
    poetry run python  -c "
import pandas as pd
df = pd.read_csv('$DATA_DIR/predictions.csv')
print(f'Total de predicciones: {len(df)}')
print(f'Min value: \${df[\"account_value\"].min():,.2f}')
print(f'Max value: \${df[\"account_value\"].max():,.2f}')
print(f'Mean value: \${df[\"account_value\"].mean():,.2f}')
print(f'Median value: \${df[\"account_value\"].median():,.2f}')
print(f'\\First 5 predictions:')
print(df.head())
"
else
    echo "File not found"
    exit 1
fi

echo ""
echo "Pipeline completed successfully!"
echo "Generated files:"
echo "   - $DATA_DIR/accounts_test_processed.csv"
echo "   - $DATA_DIR/features_test.csv" 
echo "   - $DATA_DIR/predictions.csv"
echo ""
echo "To run with Airflow:"
echo "   1. docker-compose up -d"
echo "   2. Go to http://localhost:8080 (admin/admin)"
echo "   3. Enable and execute the DAG 'coverwallet_ml_pipeline'"