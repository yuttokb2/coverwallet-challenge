#!/usr/bin/env python3
"""
Script simple de predicción para CoverWallet.
Carga el modelo y genera predicciones para datos de test.
"""

import pandas as pd
import numpy as np
import joblib
import argparse
from pathlib import Path
from utils import detect_base_path


def main():
    # Argumentos del script
    parser = argparse.ArgumentParser(description="Predicción con modelo XGBoost")
    parser.add_argument("--model-path", default="model/xgboost_model.joblib", help="Ruta al modelo")
    parser.add_argument("--features-file", default="features_test.csv", help="Archivo de features")
    parser.add_argument("--output-file", default="predictions.csv", help="Archivo de salida")
    parser.add_argument("--base-dir", default=".", help="Directorio base")
    
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(args.base_dir)
    if args.base_dir == ".":
        base_dir = detect_base_path()
    else:
        base_dir = Path(args.base_dir)

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        # Si no es absoluta, buscar desde base_dir
        if (base_dir / args.model_path).exists():
            model_path = base_dir / args.model_path
        elif (base_dir / 'model' / Path(args.model_path).name).exists():
            model_path = base_dir / 'model' / Path(args.model_path).name

    features_path = base_dir / "data" / args.features_file
    output_path = base_dir / "data" / args.output_file
    
    print(f"Model: {model_path}")
    print(f"Features: {features_path}")
    print(f"Output: {output_path}")
    

    print("Loading model...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")
    

    print("Loading features...")
    df = pd.read_csv(features_path)
    print(f"Features loaded: {df.shape}")
    

    print("🔧 Preparando datos...")
    
    # Guardar account_uuid
    account_uuids = df['account_uuid'].copy()
    
    feature_names = model.feature_names_in_ 

    
    # Seleccionar solo las features necesarias
    X = df[feature_names].copy()
    
    # In case there are null values (there shouldn't be any if preprocessing was done correctly)
    if X.isnull().any().any():
        print("Filling null values...")
        X = X.fillna(X.median())
    

    
    print("Predicting...")
    predictions = model.predict(X)
    
    
    results_df = pd.DataFrame({
        'account_uuid': account_uuids,
        'account_value': predictions.round(2)
    })
    
    results_df.to_csv(output_path, index=False)
    print(f"Results saved in: {output_path}")
    
    print("\nPredictions stats:")
    print(f"   Total: {len(results_df)}")
    print(f"   Min: ${results_df['account_value'].min():,.2f}")
    print(f"   Max: ${results_df['account_value'].max():,.2f}")
    print(f"   Mean: ${results_df['account_value'].mean():,.2f}")
    print(f"   Median: ${results_df['account_value'].median():,.2f}")
    
    print("\n Prediction completed successfully ✅")

if __name__ == "__main__":
    main()
