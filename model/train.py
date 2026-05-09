from pathlib import Path
import json
import joblib

from sklearn.ensemble import RandomForestRegressor

def preprocess_weather_features(daily_data):
    df = daily_data.copy()

    df["y"] = df["temperature_2m"]

    for lag in range(1, 9):
        df[f"temp_lag_{lag}"] = df["temperature_2m"].shift(lag)
        df[f"hum_lag_{lag}"] = df["humidity"].shift(lag)

    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month

    df = df.dropna()

    features = (
        [f"temp_lag_{lag}" for lag in range(1, 9)] +
        [f"hum_lag_{lag}" for lag in range(1, 9)] +
        ["hour", "dayofweek", "month"]
    )

    X = df[features]
    y = df["y"]

    return X, y, features


def train_weather_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model


def save_weather_model(model, X_train, X_val, X_test, features):
    registry_dir = Path("../model/registry")
    registry_dir.mkdir(parents=True, exist_ok=True)

    model_version = "weather_rf_v1"

    model_path = registry_dir / f"{model_version}.joblib"
    metadata_path = registry_dir / f"{model_version}_metadata.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_version": model_version,
        "model_type": "RandomForestRegressor",
        "target": "temperature_2m",
        "frequency": "3h",
        "features": features,
        "training_period": {
            "start": str(X_train.index.min()),
            "end": str(X_train.index.max())
        },
        "validation_period": {
            "start": str(X_val.index.min()),
            "end": str(X_val.index.max())
        },
        "test_period": {
            "start": str(X_test.index.min()),
            "end": str(X_test.index.max())
        }
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)

    print("Model saved:", model_path)
    print("Metadata saved:", metadata_path)


def train_pipeline(daily_data):
    X, y, features = preprocess_weather_features(daily_data)

    n = len(X)
    n_train = int(n * 0.65)
    n_val = int(n * 0.20)

    X_train = X.iloc[:n_train]
    y_train = y.iloc[:n_train]

    X_val = X.iloc[n_train:n_train + n_val]

    X_test = X.iloc[n_train + n_val:]

    model = train_weather_model(X_train, y_train)

    save_weather_model(model, X_train, X_val, X_test, features)

    return model
