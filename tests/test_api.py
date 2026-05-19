import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


VALID_PAYLOAD = {
    "temp_lag_1": 12.0,
    "temp_lag_2": 12.5,
    "temp_lag_3": 13.0,
    "temp_lag_4": 13.5,
    "temp_lag_5": 14.0,
    "temp_lag_6": 14.5,
    "temp_lag_7": 15.0,
    "temp_lag_8": 15.5,
    "hum_lag_1": 70.0,
    "hum_lag_2": 71.0,
    "hum_lag_3": 72.0,
    "hum_lag_4": 73.0,
    "hum_lag_5": 74.0,
    "hum_lag_6": 75.0,
    "hum_lag_7": 76.0,
    "hum_lag_8": 77.0,
    "hour": 12,
    "dayofweek": 2,
    "month": 5,
}


class DummyModel:
    def predict(self, X):
        return [18.5 for _ in range(len(X))]


@pytest.fixture()
def client(monkeypatch, tmp_path):
    import common
    import model.weather_model as weather_model

    metadata_path = tmp_path / "weather_rf_v1_metadata.json"
    metadata_path.write_text(
        '{"model_version": "weather_rf_v1_test"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(common, "METADATA_PATH", metadata_path)
    monkeypatch.setattr(common, "DB_PATH", tmp_path / "test_weather.db")

    monkeypatch.setattr(
        weather_model,
        "load_weather_model",
        lambda: DummyModel(),
    )

    sys.modules.pop("api.main", None)
    api_main = importlib.import_module("api.main")

    return TestClient(api_main.app)


def test_root_endpoint_returns_api_status(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Weather Prediction API is running"}


def test_version_endpoint_returns_string(client):
    response = client.get("/version")

    assert response.status_code == 200
    assert isinstance(response.json(), str)
    assert response.json() != ""


def test_predict_endpoint_returns_prediction_and_model_version(client):
    response = client.post("/predict", json=VALID_PAYLOAD)

    assert response.status_code == 200

    data = response.json()
    assert "predicted_temperature_2m" in data
    assert "model_version" in data
    assert isinstance(data["predicted_temperature_2m"], float)
    assert data["predicted_temperature_2m"] == 18.5


def test_predict_endpoint_rejects_invalid_hour(client):
    payload = VALID_PAYLOAD.copy()
    payload["hour"] = 30

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert "hour invalide" in response.json()["detail"]


def test_predict_endpoint_rejects_invalid_humidity(client):
    payload = VALID_PAYLOAD.copy()
    payload["hum_lag_1"] = 150.0

    response = client.post("/predict", json=payload)

    assert response.status_code == 400
    assert "humidity invalide" in response.json()["detail"]


def test_predict_batch_endpoint_returns_one_prediction_per_item(client):
    response = client.post(
        "/predict_batch",
        json=[VALID_PAYLOAD, VALID_PAYLOAD],
    )

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2

    for item in data:
        assert item["predicted_temperature_2m"] == 18.5
        assert "model_version" in item


def test_predictions_endpoint_returns_saved_predictions(client):
    client.post("/predict", json=VALID_PAYLOAD)

    response = client.get("/predictions")

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["predicted_temperature_2m"] == 18.5
    assert "model_version" in data[0]
    assert "inference_at" in data[0]
    
    