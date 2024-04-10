from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

model = joblib.load("model.pkl")

class InputData(BaseModel):
    N: float
    P: float
    K: float

@app.post("/recommend/")
async def recommend_crop(data: InputData):
    try:
        # Extract features from request body
        features = np.array([[data.N, data.P, data.K]])
        # Perform prediction using the pre-trained model
        recommendation = model.predict(features)[0]
        # Return the recommendation
        return {"recommendation": recommendation}
    except Exception as e:
        # In case of any error, return an HTTP 500 error response
        raise HTTPException(status_code=500, detail=str(e))

#To Run
    
#pip install numpy joblib uvicorn lightgbm
#uvicorn main:app --reload --port 8000





