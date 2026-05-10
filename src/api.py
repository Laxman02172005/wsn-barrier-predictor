from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import predict, load_artifacts
from src.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="WSN Barrier Prediction API")

class WSNInput(BaseModel):
    Area: float
    Sensing_Range: float
    Transmission_Range: float
    Number_of_Sensor_nodes: float

class PredictionOutput(BaseModel):
    predicted_barriers: float

@app.on_event("startup")
async def startup_event():
    try:
        load_artifacts()
        logger.info("API successfully loaded artifacts on startup.")
    except Exception as e:
        logger.error(f"Failed to load artifacts on startup: {e}")

@app.get("/health")
def health_check():
    """Health check endpoint for Kubernetes/Docker."""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionOutput)
def make_prediction(data: WSNInput):
    """Predicts Number of Barriers based on WSN features."""
    try:
        # Convert pydantic model to dict, fixing names to match training data
        input_dict = {
            "Area": data.Area,
            "Sensing Range": data.Sensing_Range,
            "Transmission Range": data.Transmission_Range,
            "Number of Sensor nodes": data.Number_of_Sensor_nodes
        }
        
        prediction = predict(input_dict)
        return PredictionOutput(predicted_barriers=prediction)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
