# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import json
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware



app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load model and ideal conditions
model = joblib.load("model.pkl")
with open("ideal_conditions.json", "r") as f:
    ideal_data = json.load(f)

# Define request schema
class SensorData(BaseModel):
    temperature: float
    nitrogen: float
    phosphorus: float
    potassium: float
    humidity: float
    pH_Value: float
    rainfall: float
@app.post("/predict_crop")
def predict_crop(data: SensorData):
    print("👉 Incoming data:", data.dict())
    
    # Store sensor data for response
    sensor_data = {
        "humidity": data.humidity,
        "nitrogen": data.nitrogen,
        "pH_Value": data.pH_Value,
        "phosphorus": data.phosphorus,
        "potassium": data.potassium,
        "rainfall": data.rainfall,
        "temperature": data.temperature
    }
    
    input_data = np.array([[data.temperature, data.nitrogen, data.phosphorus,
                            data.potassium, data.humidity, data.pH_Value, data.rainfall]])

    try:
        prediction = model.predict(input_data)[0]
        print("✅ Predicted crop:", prediction)
    except Exception as e:
        print("❌ Model prediction failed:", e)
        raise

    ideal = ideal_data.get(prediction, {})
    # Reorder ideal conditions to match required order
    ordered_ideal = {
        "Humidity": ideal["Humidity"],
        "Nitrogen": ideal["Nitrogen"],
        "Phosphorus": ideal["Phosphorus"],
        "Potassium": ideal["Potassium"],
        "Rainfall": ideal["Rainfall"],
        "Temperature": ideal["Temperature"],
        "pH_Value": ideal["pH_Value"]
    }
    
    suggestions = {}
    # Process suggestions in the required order
    for key in ["humidity", "nitrogen", "ph_value", "phosphorus", "potassium", "rainfall", "temperature"]:
        ideal_key = key.capitalize() if key != "ph_value" else "pH_Value"
        ideal_val = ideal[ideal_key]
        actual = sensor_data[ideal_key] if ideal_key == "pH_Value" else sensor_data[key]

        if actual is None:
            suggestions[key] = f"Missing value for {key}"
        elif abs(actual - ideal_val) < 1:
            suggestions[key] = "Optimal"
        elif actual < ideal_val:
            suggestions[key] = f"Increase by {round(ideal_val - actual, 2)}"
        else:
            suggestions[key] = f"Decrease by {round(actual - ideal_val, 2)}"

    return {
        "ideal_conditions": ordered_ideal,
        "predicted_crop": prediction,
        "sensor_data": sensor_data,
        "suggestions": suggestions
    }

class PartialSensorData(BaseModel):
    crop: str
    temperature: Optional[float]
    nitrogen: Optional[float]
    phosphorus: Optional[float]
    potassium: Optional[float]
    humidity: Optional[float]
    rainfall: Optional[float]
    pH_Value: Optional[float] = None
@app.post("/complete_conditions")
def complete_conditions(data: PartialSensorData):
    crop = data.crop.strip()

    if crop not in ideal_data:
        return {"error": f"Crop '{crop}' not found in ideal conditions database."}

    ideal = ideal_data[crop]
    suggestions = {}

    for key, ideal_val in ideal.items():
        # Convert both keys to lowercase for comparison
        key_lower = key.lower()
        # Get the value from data using case-insensitive matching
        actual = None
        for data_key, data_val in data.dict().items():
            if data_key.lower() == key_lower:
                actual = data_val
                break

        if actual is None:
            suggestions[key] = f"Missing. Ideal is {ideal_val}"
        elif abs(actual - ideal_val) < 1:
            suggestions[key] = "Optimal"
        elif actual < ideal_val:
            suggestions[key] = f"Increase by {round(ideal_val - actual, 2)}"
        else:
            suggestions[key] = f"Decrease by {round(actual - ideal_val, 2)}"

    return {
        "crop": crop,
        "ideal_conditions": ideal,
        "suggestions": suggestions
    }
