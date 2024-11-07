from fastapi import FastAPI
from typing import List
from pydantic import BaseModel
app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Serwer API działa poprawnie"}


class PredictionResponse(BaseModel):
    labels: List[str]


@app.post("/predict_siamase", response_model=PredictionResponse)
def predict_siamase():
    # pobieranie obrazu
    # wykonanie predykcji

    labels = ["produkt1", "produkt2", "produkt3"]

    return PredictionResponse(labels=labels)



