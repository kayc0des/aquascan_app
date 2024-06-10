from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Mean and standard deviation for normalization
mean = np.array([7.08079450e+00, 1.96369496e+02, 2.20140925e+04, 7.12227679e+00,
                 3.33775777e+02, 4.26205111e+02, 1.42849702e+01, 6.63962929e+01,
                 3.96678617e+00])
std = np.array([1.46973160e+00, 3.28747428e+01, 8.76723242e+03, 1.58284325e+00,
                3.61370955e+01, 8.08117273e+01, 3.30765705e+00, 1.57674742e+01,
                7.80263293e-01])

class WaterQualityInput(BaseModel):
    ph: float
    Hardness: float
    Solids: float
    Chloramines: float
    Sulfate: float
    Conductivity: float
    Organic_carbon: float
    Trihalomethanes: float
    Turbidity: float

@app.post("/predict")
async def predict(data: WaterQualityInput):
    # Convert input data to numpy array
    input_data = np.array([[
        data.ph, data.Hardness, data.Solids, data.Chloramines, 
        data.Sulfate, data.Conductivity, data.Organic_carbon, 
        data.Trihalomethanes, data.Turbidity
    ]])
    
    # Normalize thse input data
    input_data = (input_data - mean) / std
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Return the prediction
    return {"prediction": float(prediction[0][0])}
