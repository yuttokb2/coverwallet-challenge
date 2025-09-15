#!/bin/bash
# Script para probar el pipeline completo de ML

set -e

echo "🚀 Iniciando prueba del pipeline ML de CoverWallet..."

# Variables
BASE_DIR=$(pwd)
DATA_DIR="$BASE_DIR/data"
SRC_DIR="$BASE_DIR/src"
MODEL_DIR="$BASE_DIR/model"

echo "📁 Verificando archivos necesarios..."

# Verificar archivos de entrada
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
echo "🔄 Ejecutando pipeline paso a paso..."

# Paso 1: Preprocessing
echo "1️⃣ Ejecutando preprocessing..."
poetry run python src/preprocessing_data.py --config src/config/config_preprocess.yaml --dataset test
if [ $? -eq 0 ]; then
    echo "✅ Preprocessing completado"
else
    echo "❌ Error en preprocessing"
    exit 1
fi

# Paso 2: Wrangling/Feature Engineering
echo "2️⃣ Ejecutando wrangling (feature engineering)..."
poetry run python src/wrangling.py --config src/config/config_wrangling.yaml --dataset test
if [ $? -eq 0 ]; then
    echo "✅ Wrangling completado"
else
    echo "❌ Error en wrangling"
    exit 1
fi

# Paso 3: Predicción
echo "3️⃣ Ejecutando predicción..."
poetry run python src/detect.py --model-path model/xgboost_model.joblib --features-file features_test.csv --output-file predictions.csv
if [ $? -eq 0 ]; then
    echo "✅ Predicción completada"
else
    echo "❌ Error en predicción"
    exit 1
fi

# Verificar resultados
echo ""
echo "📊 Verificando resultados..."

if [ -f "$DATA_DIR/predictions.csv" ]; then
    echo "✅ Archivo de predicciones generado: $DATA_DIR/predictions.csv"
    
    # Mostrar estadísticas básicas
    echo ""
    echo "📈 Estadísticas de predicciones:"
    python3 -c "
import pandas as pd
df = pd.read_csv('$DATA_DIR/predictions.csv')
print(f'Total de predicciones: {len(df)}')
print(f'Valor mínimo: \${df[\"account_value\"].min():,.2f}')
print(f'Valor máximo: \${df[\"account_value\"].max():,.2f}')
print(f'Valor promedio: \${df[\"account_value\"].mean():,.2f}')
print(f'Valor mediano: \${df[\"account_value\"].median():,.2f}')
print(f'\\nPrimeras 5 predicciones:')
print(df.head())
"
else
    echo "❌ Archivo de predicciones no encontrado"
    exit 1
fi

echo ""
echo "🎉 Pipeline completado exitosamente!"
echo "📁 Archivos generados:"
echo "   - $DATA_DIR/accounts_test_processed.csv"
echo "   - $DATA_DIR/features_test.csv" 
echo "   - $DATA_DIR/predictions.csv"
echo ""
echo "🚀 Para ejecutar con Airflow:"
echo "   1. docker-compose up -d"
echo "   2. Ir a http://localhost:8080 (admin/admin)"
echo "   3. Activar y ejecutar el DAG 'coverwallet_ml_pipeline'"
