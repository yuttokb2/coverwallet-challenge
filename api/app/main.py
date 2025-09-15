from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List, Dict
from pathlib import Path

# Get the absolute path to the model
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Goes up to project root
MODEL_PATH = BASE_DIR / "model" / "xgboost_model.joblib"

app = FastAPI(
    title="XGBoost Model API",
    description="API for serving predicting account_value using a pre-trained XGBoost model",
    version="1.0.0"
)

# Load model at startup
model = None

# Features en el orden EXACTO que espera tu modelo (37 features)
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

@app.on_event("startup")
async def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

class PredictionInputDict(BaseModel):
    features: Dict[str, float]
    
    class Config:
        schema_extra = {
            "example": {
                "features": {
                    "log_total_payroll": 15.2,
                    "year_established": 2010.0,
                    "total_payroll": 4250000.0,
                    "product_concentration": 0.6,
                    "state_premium_sum_encoded": 0.9,
                    "subindustry_sum_premium_encoded": 0.6,
                    "carrier_concentration": 0.55,
                    "business_structure_revenue_encoded": 0.5,
                    "premium_per_employee": 2.5,
                    "industry_revenue_encoded": 0.75,
                    "num_quotes": 12.0,
                    "revenue_x_payroll": 12750000.0,
                    "log_annual_revenue": 16.5,
                    "premium_to_revenue_ratio": 0.12,
                    "total_quotes": 85.0,
                    "premium_ratio_max_avg": 2.8,
                    "annual_revenue": 15000000.0,
                    "max_x_nquotes": 144.0,
                    "state_revenue_encoded": 0.6,
                    "num_employees": 50.0,
                    "business_structure_premium_sum_encoded": 0.8,
                    "industry_sum_premium_encoded": 0.7,
                    "num_products_requested": 3.0,
                    "iqr_premium": 3000.0,
                    "premium_per_revenue": 0.15,
                    "avg_x_nproducts": 25.5,
                    "subindustry_revenue_encoded": 0.65,
                    "premium_per_quote": 1200.0,
                    "max_premium": 25000.0,
                    "quotes_per_million_revenue": 0.08,
                    "avg_premium": 8500.0,
                    "sum_premium": 102000.0,
                    "carrier_diversity": 0.4,
                    "quotes_per_employee": 0.15,
                    "min_premium": 500.0,
                    "num_carriers": 4.0,
                    "revenue_per_employee": 85000.0
                }
            }
        }

class PredictionOutput(BaseModel):
    prediction: float
    status: str

@app.get("/")
async def root():
    return {"message": "XGBoost Model API", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "required_features": len(feature_names)
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_dict(input: PredictionInputDict):
    try:
        # Validar que todas las features estén presentes
        missing_features = []
        for feature_name in feature_names:
            if feature_name not in input.features:
                missing_features.append(feature_name)
        
        if missing_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Missing features: {missing_features}"
            )
        
        # Convert dictionary to a list ordered by feature_names
        features_list = []
        for feature_name in feature_names:
            features_list.append(input.features[feature_name])
        
        features_array = np.array(features_list).reshape(1, -1)
        
        # Validating that the array has the correct number of features
        if features_array.shape[1] != len(feature_names):
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {len(feature_names)} features, got {features_array.shape[1]}"
            )
        
        prediction = model.predict(features_array)
        
        return {
            "prediction": float(prediction[0]),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/feature-names")
async def get_feature_names():
    return {
        "feature_names": feature_names,
        "total_features": len(feature_names)
    }

@app.get("/model-info")
async def model_info():
    if hasattr(model, 'get_params'):
        return {"model_params": model.get_params()}
    return {"message": "Model parameters not available"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)