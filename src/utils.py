import os
import json
import joblib
from src.logger import get_logger

logger = get_logger(__name__)

def create_directory(path: str):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"Created directory: {path}")

def save_object(obj, path: str):
    """Saves a python object using joblib."""
    try:
        joblib.dump(obj, path)
        logger.info(f"Saved object to {path}")
    except Exception as e:
        logger.error(f"Error saving object to {path}: {e}")
        raise e

def load_object(path: str):
    """Loads a python object using joblib."""
    try:
        obj = joblib.load(path)
        logger.info(f"Loaded object from {path}")
        return obj
    except Exception as e:
        logger.error(f"Error loading object from {path}: {e}")
        raise e

def save_json(data: dict, path: str):
    """Saves a dictionary to a JSON file."""
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved JSON to {path}")
    except Exception as e:
        logger.error(f"Error saving JSON to {path}: {e}")
        raise e

def load_json(path: str) -> dict:
    """Loads a dictionary from a JSON file."""
    try:
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded JSON from {path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {path}: {e}")
        raise e
