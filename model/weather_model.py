from pathlib import Path
import joblib


MODEL_PATH = Path(__file__).parent / "registry" / "weather_rf_v1.joblib"


def load_weather_model():
    model = joblib.load(MODEL_PATH)
    return model


def predict_temperature(model, X):
    predictions = model.predict(X)
    return predictions
