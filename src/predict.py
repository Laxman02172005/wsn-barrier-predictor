import pandas as pd
import numpy as np
from src.config import MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH
from src.utils import load_object, load_json
from src.preprocess import feature_engineering
from src.logger import get_logger

logger = get_logger(__name__)

# Lazy loading of artifacts
_model = None
_scaler = None
_feature_columns = None

def load_artifacts():
    global _model, _scaler, _feature_columns
    if _model is None:
        logger.info("Loading model artifacts...")
        _model = load_object(MODEL_PATH)
        _scaler = load_object(SCALER_PATH)
        _feature_columns = load_json(FEATURE_COLUMNS_PATH)["features"]
    return _model, _scaler, _feature_columns

def predict(input_data: dict) -> float:
    """Makes a prediction from input dictionary."""
    model, scaler, feature_columns = load_artifacts()
    
    df_input = pd.DataFrame([input_data])
    
    # Apply feature engineering
    df_input = feature_engineering(df_input)
    
    # Ensure correct column order
    df_input = df_input[feature_columns]
    
    # Scale
    X_scaled = scaler.transform(df_input)
    
    # Predict
    prediction = model.predict(X_scaled)
    return float(prediction[0])
