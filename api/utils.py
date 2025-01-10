import pandas as pd
import os
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
import joblib


PREPROCESSOR_FILE = "../models/preprocessor.pkl"
MODEL_DIR = "../models/"

cat_features = ['brand', 'model', 'origin', 'type', 'gearbox', 'fuel', 'color']
num_features = ['list_time', 'manufacture_date', 'seats', 'mileage_v2', 'Vehicle_Age', 'Mileage_per_Year', 'milage_with_age', 'Mileage_per_Year_with_age', 'Is_Luxury_Brand']

preprocessor = joblib.load(PREPROCESSOR_FILE)

catboost_models = []
lgb_models = []

for file_name in os.listdir(MODEL_DIR):
    file_path = os.path.join(MODEL_DIR, file_name)
    if file_name.startswith("catboost_model") and file_name.endswith(".joblib"):
        catboost_models.append(joblib.load(file_path))
    elif file_name.startswith("lgb_model") and file_name.endswith(".joblib"):
        lgb_models.append(joblib.load(file_path))

fitted_models = catboost_models + lgb_models

class VotingModel(BaseEstimator, RegressorMixin):
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def predict(self, dataframe):
        y_preds = [estimator.predict(dataframe) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_chunked(self, dataframe, chunk_size=1000):
        n = len(dataframe)
        chunk_preds = []
        for i in range(0, n, chunk_size):
            chunk = dataframe[i:i + chunk_size]
            y_preds_chunk = [estimator.predict(chunk) for estimator in self.estimators]
            chunk_mean = np.mean(y_preds_chunk, axis=0)
            chunk_preds.extend(chunk_mean)
        return np.array(chunk_preds)

model = VotingModel(fitted_models)


def extract_age_features(df):
    current_year = 2024

    df['Vehicle_Age'] = current_year - df['manufacture_date']
    
    df['Mileage_per_Year'] = df['mileage_v2'] / df['Vehicle_Age']
    df['milage_with_age'] =  df.groupby('Vehicle_Age')['mileage_v2'].transform('mean')
    
    df['Mileage_per_Year_with_age'] =  df.groupby('Vehicle_Age')['Mileage_per_Year'].transform('mean')
    
    return df

def extract_other_features(df):
    
    luxury_brands =  ['Mercedes Benz', 'BMW', 'Audi', 'Porsche', 'LandRover', 
                    'Lexus', 'Jaguar', 'Bentley', 'Maserati', 'Lamborghini', 
                    'Rolls Royce', 'Ferrari', 'Aston Martin', 'Maybach']
    df['Is_Luxury_Brand'] = df['brand'].apply(lambda x: 1 if x in luxury_brands else 0)
    return df
