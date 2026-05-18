"""
Ce script doit contenir l'implémentation des endpoints pour les fonctionnalités suivantes :
- Génération des prédictions pour une date donnée,
- Récupération des prédictions pour une date donnée,
- Récupération des prédictions combinées avec des données réelles observées pour une période donnée
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import json
import sqlite3
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import common
from model.weather_model import load_weather_model, predict_temperature
import logging


common.APP_VERSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Charger le modèle au démarrage de l'API ---
model = load_weather_model()


# --- Lire la version du modèle depuis les métadonnées ---
with open(common.METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model_version = metadata["model_version"]
print(f"Version du modèle chargé : {model_version}")


app = FastAPI(title="Weather Prediction API")


# --- Schéma des données d'entrée ---
class WeatherInput(BaseModel):
    temp_lag_1: float
    temp_lag_2: float
    temp_lag_3: float
    temp_lag_4: float
    temp_lag_5: float
    temp_lag_6: float
    temp_lag_7: float
    temp_lag_8: float

    hum_lag_1: float
    hum_lag_2: float
    hum_lag_3: float
    hum_lag_4: float
    hum_lag_5: float
    hum_lag_6: float
    hum_lag_7: float
    hum_lag_8: float

    hour: int
    dayofweek: int
    month: int


# --- Validation des données météo ---
def validate_weather_input(data: WeatherInput):
    if not (0 <= data.hour <= 23):
        raise HTTPException(status_code=400, detail="hour invalide, attendu entre 0 et 23")

    if not (0 <= data.dayofweek <= 6):
        raise HTTPException(status_code=400, detail="dayofweek invalide, attendu entre 0 et 6")

    if not (1 <= data.month <= 12):
        raise HTTPException(status_code=400, detail="month invalide, attendu entre 1 et 12")

    humidity_values = [
        data.hum_lag_1,
        data.hum_lag_2,
        data.hum_lag_3,
        data.hum_lag_4,
        data.hum_lag_5,
        data.hum_lag_6,
        data.hum_lag_7,
        data.hum_lag_8,
    ]

    for humidity in humidity_values:
        if not (0 <= humidity <= 100):
            raise HTTPException(status_code=400, detail="humidity invalide, attendu entre 0 et 100")


# --- Sauvegarder la prédiction dans la base ---
def save_prediction(data: WeatherInput, predicted_temperature: float):
    row = data.model_dump()

    row["predicted_temperature_2m"] = predicted_temperature
    row["inference_at"] = datetime.now().isoformat()
    row["model_version"] = model_version

    with sqlite3.connect(common.DB_PATH) as con:
        pd.DataFrame([row]).to_sql(
            name="predictions",
            con=con,
            if_exists="append",
            index=False
        )


# --- Endpoint racine ---
@app.get("/")
def root():
    logger.info("Endpoint / appelé")
    return {"message": "Weather Prediction API is running"}


# --- Endpoint /predict ---
@app.post("/predict")
def predict(data: WeatherInput):
    logger.info("Endpoint /predict appelé")

    try:
        validate_weather_input(data)

        X = pd.DataFrame([data.model_dump()])
        prediction = predict_temperature(model, X)
        predicted_temperature = float(prediction[0])

        save_prediction(data, predicted_temperature)

        return {
            "predicted_temperature_2m": predicted_temperature,
            "model_version": model_version
        }

    except HTTPException:
        logger.warning("Erreur de validation sur /predict", exc_info=True)
        raise

    except Exception:
        logger.error("Erreur inattendue sur /predict", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne pendant la prédiction")


# --- Endpoint /predictions : lire les prédictions sauvegardées ---
@app.get("/predictions")
def get_predictions():
    logger.info("Endpoint /predictions appelé")

    try:
        with sqlite3.connect(common.DB_PATH) as con:
            data = pd.read_sql("SELECT * FROM predictions", con)

        logger.info("Prédictions récupérées avec succès")

        return data.to_dict(orient="records")

    except Exception:
        logger.error("Erreur lors de la récupération des prédictions", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur lors de la récupération des prédictions")



# --- Endpoint /predict_batch : prédire plusieurs exemples en une fois ---
@app.post("/predict_batch")
def predict_batch(items: list[WeatherInput]):
    logger.info("Endpoint /predict_batch appelé avec %s éléments", len(items))

    try:
        results = []

        for data in items:
            validate_weather_input(data)

            X = pd.DataFrame([data.model_dump()])
            prediction = predict_temperature(model, X)
            predicted_temperature = float(prediction[0])

            save_prediction(data, predicted_temperature)

            results.append({
                "predicted_temperature_2m": predicted_temperature,
                "model_version": model_version
            })

        logger.info("Prédictions batch générées avec succès")

        return results

    except HTTPException:
        logger.warning("Erreur de validation sur /predict_batch", exc_info=True)
        raise

    except Exception:
        logger.error("Erreur inattendue sur /predict_batch", exc_info=True)
        raise HTTPException(status_code=500, detail="Erreur interne pendant la prédiction batch")

@app.get("/version")
def get_version():
    logger.info("Endpoint /version appelé")
    return common.APP_VERSION

 
