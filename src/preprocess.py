import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import BASE_FEATURES, TARGET_COL
from src.logger import get_logger

logger = get_logger(__name__)

def load_data(data_path: str) -> pd.DataFrame:
    """Loads dataset from a CSV file."""
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Successfully loaded data from {data_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        raise e

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Applies feature engineering to the dataset."""
    logger.info("Starting feature engineering...")
    df_engineered = df.copy()
    
    # Sensor Density = Sensor Nodes / Area
    df_engineered['Sensor Density'] = df_engineered['Number of Sensor nodes'] / df_engineered['Area']
    
    # Coverage Ratio = Transmission Range / Sensing Range
    # Handle division by zero just in case
    df_engineered['Coverage Ratio'] = df_engineered.apply(
        lambda row: row['Transmission Range'] / row['Sensing Range'] if row['Sensing Range'] > 0 else 0,
        axis=1
    )
    
    # Communication Efficiency = (Transmission Range * Sensor Nodes) / Area
    df_engineered['Communication Efficiency'] = (df_engineered['Transmission Range'] * df_engineered['Number of Sensor nodes']) / df_engineered['Area']
    
    logger.info(f"Engineered features added. New shape: {df_engineered.shape}")
    return df_engineered

def preprocess_data(df: pd.DataFrame):
    """Preprocesses the data, including feature engineering and scaling."""
    logger.info("Starting data preprocessing...")
    
    # Feature Engineering
    df = feature_engineering(df)
    
    # Handle missing values by dropping them
    initial_shape = df.shape
    df = df.dropna()
    if df.shape != initial_shape:
        logger.info(f"Dropped missing values. Shape changed from {initial_shape} to {df.shape}")
        
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    feature_columns = list(X.columns)
    
    return X, y, feature_columns

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits data into train and test sets."""
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    logger.info(f"Train shapes: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test shapes: X={X_test.shape}, y={y_test.shape}")
    
    return X_train, X_test, y_train, y_test
