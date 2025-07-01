# src/api/pydantic_models.py

from pydantic import BaseModel
from typing import List, Dict, Any

class PredictionRequest(BaseModel):
    """
    Represents the input data for a prediction request.
    Assumes features are already preprocessed into a numerical vector.
    """
    features: List[float]

class PredictionResponse(BaseModel):
    """
    Represents the output data for a prediction response.
    """
    risk_probability: float
    is_high_risk: int # The binary classification (0 or 1)
    message: str = "Prediction successful."

class HealthCheckResponse(BaseModel):
    """
    Response model for the health check endpoint.
    """
    status: str = "ok"
    model_loaded: bool
    message: str