
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
import tensorflow as tf

app = FastAPI()

# Defin# Define paths
MODEL_PATH = r"C:/MLOPS_PROJECT/course-project-AaimlikAbbasi/mlruns/116437604931868982/9d429f039dae4868821ba97888cf372c/artifacts/lstm_model.h5"
SCALER_PATH = r"C:/MLOPS_PROJECT/course-project-AaimlikAbbasi/mlruns/116437604931868982/9d429f039dae4868821ba97888cf372c/artifacts/scaler.pkl"

# Validate paths
model = tf.keras.models.load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Define input model
class PredictRequest(BaseModel):
    aqi_values: list

@app.post("/predict")
def predict(request: PredictRequest):
    """
    Predict the next AQI value based on the last 10 AQI values.

    :param request: Input JSON containing aqi_values.
    :return: Predicted AQI value.
    """
    try:
        aqi_values = request.aqi_values
        # Validate input length
        if len(aqi_values) != 10:
            raise HTTPException(status_code=400, detail="Please provide exactly 10 AQI values.")

        # Convert input to numpy array
        data = np.array(aqi_values).reshape(-1, 1)

        # Scale the data using the pre-fitted scaler
        data_scaled = scaler.transform(data)

        # Prepare the input for the LSTM model
        X_input = data_scaled.reshape(1, -1, 1)

        # Make a prediction
        prediction_scaled = model.predict(X_input)

        # Inverse scale the prediction
        prediction = scaler.inverse_transform(prediction_scaled)

        return {"predicted_AQI": float(prediction[0][0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")