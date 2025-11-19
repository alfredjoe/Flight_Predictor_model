from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

# Get the directory where server.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load files from the same directory as server.py
model = joblib.load(os.path.join(BASE_DIR, 'flight_predictor_lr.joblib'))
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.joblib'))
label_encoders = joblib.load(os.path.join(BASE_DIR, 'label_encoders.joblib'))

app = FastAPI()

class FlightData(BaseModel):
    DayOfWeek: int
    Origin: str
    Dest: str
    DepDelay: float
    Month: int
    Day: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Flight Delay Predictor API!"}

@app.post("/predict/")
def predict_delay(data: FlightData):
    is_weekend = 1 if data.DayOfWeek in [1, 7] else 0
    dep_del15 = 1 if data.DepDelay > 15 else 0
    
    origin_encoded = label_encoders['Origin'].transform([data.Origin])[0]
    dest_encoded = label_encoders['Dest'].transform([data.Dest])[0]
    
    features = np.array([[
        data.DayOfWeek,
        origin_encoded,
        dest_encoded,
        data.DepDelay,
        is_weekend,
        data.Month,
        data.Day,
        dep_del15
    ]])
    
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    predicted_class = "Delay" if prediction[0] == 1 else "No Delay"
    
    return {"predicted_delay": predicted_class}
