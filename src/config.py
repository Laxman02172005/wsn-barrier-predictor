import os
from dotenv import load_dotenv

load_dotenv()

# Data configuration
DATA_PATH = os.getenv("DATA_PATH", "data/data.csv")

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "WSN_Barrier_Prediction"

# Model paths
MODEL_DIR = os.getenv("MODEL_DIR", "models/")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, "feature_columns.json")

# Features
TARGET_COL = "Number of Barriers"
BASE_FEATURES = [
    "Area",
    "Sensing Range",
    "Transmission Range",
    "Number of Sensor nodes"
]

ENGINEERED_FEATURES = [
    "Sensor Density",
    "Coverage Ratio",
    "Communication Efficiency"
]

# Model Hyperparameters
RF_PARAMS = {
    "n_estimators": 100,
    "max_depth": 10,
    "random_state": 42
}

GB_PARAMS = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": 42
}

LOG_DIR = os.getenv("LOG_DIR", "logs/")
