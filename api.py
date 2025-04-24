from fastapi import FastAPI, Query
from pydantic import BaseModel
from predictor import calculate_ideal_restocking
from typing import Optional

app = FastAPI()

class PredictionRequest(BaseModel):
    date: str
    num_outpatient_visits: int
    num_emergency_visits: int
    num_inpatient_visits: int
    quantity_of_medicine_consumed: int
    quantity_of_medicine_in_stock_remaining: int

class PredictionResponse(BaseModel):
    ideal_restocking_quantity: int

@app.get("/")
def read_root():
    return {"message": "Hospital Restocking API is live!"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: PredictionRequest):
    restocking_quantity = calculate_ideal_restocking(data.dict())
    return {"ideal_restocking_quantity": restocking_quantity}
