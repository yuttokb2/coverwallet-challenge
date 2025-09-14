import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor


BASE_DIR = Path.cwd().parent


def read_data():

    accounts_test = pd.read_csv(f'{BASE_DIR}/data/accounts_test.csv',sep=',')
    accounts_train = pd.read_csv(f'{BASE_DIR}/data/accounts_train_proccessed.csv',sep=',')
    
    quotes_test = pd.read_csv(f'{BASE_DIR}/data/quotes_test.csv',sep=',')
    quotes_train = pd.read_csv(f'{BASE_DIR}/data/quotes_train.csv',sep=',')

    return accounts_train,accounts_test, quotes_train, accounts_train

def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """Evalúa un modelo y retorna métricas"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    # y_pred = np.expm1(y_pred) 
    # y_val = np.expm1(y_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mae = mean_absolute_error(y_val, y_pred)
    r2 = r2_score(y_val, y_pred)
    
    print(f"\n{model_name}:")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    return model, rmse, y_pred

def select_features_for_model(features_df, target_column=None):
    """
    Selecciona las features más relevantes para el modelo
    Excluye columnas no numéricas y identificadores
    """
    
    # Columnas a excluir
    exclude_cols = [
        'account_uuid', 
        'state', 
        'industry', 
        'subindustry', 
        'business_structure',
        'company_size',
        'revenue_category',
        'company_age_group',
        'favorite_carrier_id'
    ]
    
    # Si hay columna target, también excluirla
    if target_column:
        exclude_cols.append(target_column)
    
    # Seleccionar solo columnas numéricas
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remover columnas excluidas
    model_features = [col for col in numeric_cols if col not in exclude_cols]
    
    return model_features
