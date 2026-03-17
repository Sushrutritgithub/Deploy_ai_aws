from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

model = joblib.load('model.pkl')

@app.get("/")
def home():
    return {"message":"Model is running"}

@app.post("/predict")
def predict(data: list):
    prediction = model.predict(np.array(data).reshape(1,-1))
    return {"prediction": prediction.tolist()}