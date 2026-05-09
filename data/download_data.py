import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
import requests

import common


def download_data():
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": common.LATITUDE,
        "longitude": common.LONGITUDE,
        "start_date": common.DEFAULT_START_DATE,
        "end_date": common.DEFAULT_END_DATE,
        "hourly": "temperature_2m,relative_humidity_2m",
        "timezone": "Europe/Paris",
    }

    print("Téléchargement des données météo depuis Open-Meteo")

    response = requests.get(url, params=params)
    response.raise_for_status()

    json_data = response.json()
    hourly = json_data["hourly"]

    data = pd.DataFrame({
        "date": pd.to_datetime(hourly["time"]),
        "temperature_2m": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
    })

    data = data.dropna()

    print(f"Données téléchargées : {data.shape[0]} lignes, {data.shape[1]} colonnes")

    return data


def save_data(data):
    data_train, data_test = train_test_split(
        data,
        test_size=0.3,
        shuffle=False,
        random_state=common.RANDOM_STATE
    )

    print(f"Sauvegarde dans : {common.DB_PATH}")

    with sqlite3.connect(common.DB_PATH) as con:
        data_train.to_sql(name="train", con=con, if_exists="replace", index=False)
        data_test.to_sql(name="test", con=con, if_exists="replace", index=False)

    print(f"Train : {len(data_train)} lignes | Test : {len(data_test)} lignes")


if __name__ == "__main__":
    data = download_data()
    save_data(data)
