# src/api/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
import numpy as np
import os
import sys

# Add src directory to Python path to enable imports from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.pydantic_models import PredictionRequest, PredictionResponse, HealthCheckResponse

app = FastAPI(
    title="Credit Risk Probability API",
    description="API for predicting credit risk (high/low) based on transaction data.",
    version="1.0.0"
)

# --- MLflow Model Loading ---
# IMPORTANT: Replace 'YOUR_MLFLOW_RUN_ID_FOR_BEST_MODEL' with your actual MLflow Run ID
# Example: "runs:/a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6/logisticregression"
# Or if you logged it as a registered model: "models:/MyCreditRiskModel/Production"
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", "runs:/YOUR_MLFLOW_RUN_ID_FOR_BEST_MODEL/logisticregression")

model = None
model_loaded = False
load_error_message = ""

@app.on_event("startup")
async def load_model():
    """
    Load the MLflow model when the FastAPI application starts up.
    """
    global model, model_loaded, load_error_message
    try:
        print(f"Attempting to load model from: {MLFLOW_MODEL_URI}")
        model = mlflow.pyfunc.load_model(MLFLOW_MODEL_URI)
        model_loaded = True
        print("Model loaded successfully!")
    except Exception as e:
        load_error_message = f"Failed to load model: {e}"
        model_loaded = False
        print(load_error_message)

@app.get("/health", response_model=HealthCheckResponse, summary="Health Check")
async def health_check():
    """
    Checks the health of the API and model loading status.
    """
    return HealthCheckResponse(
        model_loaded=model_loaded,
        message="Model loaded successfully." if model_loaded else load_error_message
    )

@app.post("/predict", response_model=PredictionResponse, summary="Predict Credit Risk")
async def predict_risk(request: PredictionRequest):
    """
    Receives preprocessed features and returns the predicted credit risk probability
    and binary classification (high_risk or low_risk).

    The input `features` list must correspond to the exact order and type of
    features expected by the trained model *after* all preprocessing steps
    (feature engineering, scaling, encoding) have been applied.
    """
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Please check API health.")

    try:
        # Convert the list of features to a numpy array, reshape for single prediction
        input_data = np.array(request.features).reshape(1, -1)

        # Predict probability of the positive class (is_high_risk = 1)
        # Assuming the model's predict_proba returns probabilities for [class 0, class 1]
        risk_probability = model.predict_proba(input_data)[:, 1][0]

        # Classify as high_risk (1) or low_risk (0) based on a threshold
        # You might want to make this threshold configurable
        high_risk_threshold = 0.5
        is_high_risk = 1 if risk_probability >= high_risk_threshold else 0

        return PredictionResponse(
            risk_probability=risk_probability,
            is_high_risk=is_high_risk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")