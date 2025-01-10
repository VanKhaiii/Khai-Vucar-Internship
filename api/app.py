from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import uvicorn
from typing import List

from utils import preprocessor, model, extract_age_features, extract_other_features, cat_features, num_features

app = FastAPI()

class PredictRequest(BaseModel):
    list_time: int
    manufacture_date: int
    brand: str
    model: str
    origin: str
    type: str
    seats: float
    gearbox: str
    fuel: str
    color: str
    mileage_v2: float

@app.get("/car")
async def health_check():
    return {"status": "API is running."}

# Endpoint for predictions
@app.post("/predict")
async def predict(requests: List[PredictRequest]):
    try:
        input_data = pd.DataFrame([request.dict() for request in requests])

        input_data = extract_age_features(input_data)
        input_data = extract_other_features(input_data)

        transformed_data = preprocessor.transform(input_data)
        transformed_data = pd.DataFrame(transformed_data, columns=num_features + cat_features)
        # print(transformed_data)

        predictions = model.predict(transformed_data)

        predictions_list = predictions.tolist()

        results = {
        f"{row.dict()['brand']}_{row.dict()['model']}_{row.dict()['origin']}": price
        for row, price in zip(requests, predictions_list)
        }
        
        return {"predictions": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)