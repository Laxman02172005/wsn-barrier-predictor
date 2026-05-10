import os
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

from src.config import (
    DATA_PATH, MODEL_DIR, MODEL_PATH, SCALER_PATH, METRICS_PATH, 
    FEATURE_COLUMNS_PATH, EXPERIMENT_NAME, RF_PARAMS, GB_PARAMS
)
from src.preprocess import load_data, preprocess_data, split_data
from src.utils import create_directory, save_object, save_json
from src.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(y_true, y_pred):
    """Evaluates the model and returns metrics."""
    import numpy as np
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    return {"r2": r2, "mse": mse, "rmse": rmse, "mae": mae}

def train_models():
    """Trains multiple models and logs them to MLflow, then saves the best."""
    logger.info("Starting model training pipeline...")
    
    # 1. Load Data
    df = load_data(DATA_PATH)
    
    # 2. Preprocess Data
    X, y, feature_columns = preprocess_data(df)
    
    # 3. Split Data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # 4. Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create output dir
    create_directory(MODEL_DIR)
    
    # Save scaler and feature columns
    save_object(scaler, SCALER_PATH)
    save_json({"features": feature_columns}, FEATURE_COLUMNS_PATH)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(**RF_PARAMS),
        "Gradient Boosting Regressor": GradientBoostingRegressor(**GB_PARAMS)
    }
    
    best_model = None
    best_r2 = -float("inf")
    best_model_name = ""
    all_metrics = {}
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        with mlflow.start_run(run_name=name):
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            metrics = evaluate_model(y_test, y_pred)
            all_metrics[name] = metrics
            
            logger.info(f"{name} metrics: {metrics}")
            
            # Log params, metrics, model
            if hasattr(model, "get_params"):
                mlflow.log_params(model.get_params())
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, artifact_path="model")
            
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                best_model = model
                best_model_name = name

    logger.info(f"Best model: {best_model_name} with R2: {best_r2}")
    
    # Save the best model
    save_object(best_model, MODEL_PATH)
    
    # Save all metrics for Streamlit comparison
    save_json(all_metrics, METRICS_PATH)
    
    logger.info("Training pipeline completed successfully.")

if __name__ == "__main__":
    train_models()
