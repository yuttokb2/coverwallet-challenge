from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
from typing import List,Dict
from pathlib import Path

# Get the absolute path to the model
BASE_DIR = Path(__file__).resolve().parent.parent.parent  # Goes up to project root
MODEL_PATH = BASE_DIR / "model" / "xgboost_model.joblib"

app = FastAPI(
    title="XGBoost Model API",
    description="API for serving predicting accoun_value using a pre-trained XGBoost model",
    version="1.0.0"
)

# Load model at startup
model = None
# Load your model

feature_names = [
    'log_total_payroll', 'num_products_requested', 'business_structure_revenue_encoded',
    'business_structure_premium_sum_encoded', 'max_premium', 'premium_per_quote',
    'premium_per_revenue', 'industry_sum_premium_encoded', 'num_quotes',
    'product_concentration', 'avg_premium', 'min_premium', 'state_premium_sum_encoded',
    'premium_per_employee', 'carrier_diversity', 'iqr_premium', 'revenue_per_employee',
    'quotes_per_million_revenue', 'log_annual_revenue', 'premium_to_revenue_ratio',
    'avg_x_nproducts', 'max_x_nquotes', 'premium_ratio_max_avg', 'sum_premium',
    'quotes_per_employee', 'state_revenue_encoded', 'num_carriers', 'carrier_concentration',
    'total_quotes', 'premium_range', 'industry_revenue_encoded', 'revenue_x_payroll'
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
                    "num_products_requested": 3.0,
                    "business_structure_revenue_encoded": 0.5,
                    "business_structure_premium_sum_encoded": 0.8,
                    "max_premium": 25000.0,
                    "premium_per_quote": 1200.0,
                    "premium_per_revenue": 0.15,
                    "industry_sum_premium_encoded": 0.7,
                    "num_quotes": 12.0,
                    "product_concentration": 0.6,
                    "avg_premium": 8500.0,
                    "min_premium": 500.0,
                    "state_premium_sum_encoded": 0.9,
                    "premium_per_employee": 2.5,
                    "carrier_diversity": 0.4,
                    "iqr_premium": 3000.0,
                    "revenue_per_employee": 85000.0,
                    "quotes_per_million_revenue": 0.08,
                    "log_annual_revenue": 16.5,
                    "premium_to_revenue_ratio": 0.12,
                    "avg_x_nproducts": 25.5,
                    "max_x_nquotes": 144.0,
                    "premium_ratio_max_avg": 2.8,
                    "sum_premium": 102000.0,
                    "quotes_per_employee": 0.15,
                    "state_revenue_encoded": 0.6,
                    "num_carriers": 4.0,
                    "carrier_concentration": 0.55,
                    "total_quotes": 85.0,
                    "premium_range": 24500.0,
                    "industry_revenue_encoded": 0.75,
                    "revenue_x_payroll": 12750000.0
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
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
async def predict_dict(input: PredictionInputDict):
    try:
        # Convert dictionary to ordered list based on feature_names
        features_list = []
        for feature_name in feature_names:
            if feature_name not in input.features:
                raise HTTPException(status_code=400, detail=f"Missing feature: {feature_name}")
            features_list.append(input.features[feature_name])
        
        features_array = np.array(features_list).reshape(1, -1)
        prediction = model.predict(features_array)
        
        return {
            "prediction": float(prediction[0]),
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/feature-names")
async def get_feature_names():
    return {"feature_names": feature_names}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "model_loaded": model is not None,
        "required_features": len(feature_names)
    }


@app.get("/model-info")
async def model_info():
    if hasattr(model, 'get_params'):
        return {"model_params": model.get_params()}
    return {"message": "Model parameters not available"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)