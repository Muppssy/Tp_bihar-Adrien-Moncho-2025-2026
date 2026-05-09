import pandas as pd

from weather_model import load_weather_model, predict_temperature


def test_weather_model_inference():
    model = load_weather_model()

    examples = pd.DataFrame([
        {
            "temp_lag_1": 24.1,
            "temp_lag_2": 23.8,
            "temp_lag_3": 23.5,
            "temp_lag_4": 22.9,
            "temp_lag_5": 22.4,
            "temp_lag_6": 22.0,
            "temp_lag_7": 21.8,
            "temp_lag_8": 21.5,
            "hum_lag_1": 65,
            "hum_lag_2": 66,
            "hum_lag_3": 67,
            "hum_lag_4": 69,
            "hum_lag_5": 70,
            "hum_lag_6": 72,
            "hum_lag_7": 73,
            "hum_lag_8": 74,
            "hour": 12,
            "dayofweek": 2,
            "month": 5,
        },
        {
            "temp_lag_1": 18.2,
            "temp_lag_2": 18.0,
            "temp_lag_3": 17.7,
            "temp_lag_4": 17.5,
            "temp_lag_5": 17.1,
            "temp_lag_6": 16.9,
            "temp_lag_7": 16.5,
            "temp_lag_8": 16.2,
            "hum_lag_1": 80,
            "hum_lag_2": 81,
            "hum_lag_3": 82,
            "hum_lag_4": 83,
            "hum_lag_5": 84,
            "hum_lag_6": 85,
            "hum_lag_7": 86,
            "hum_lag_8": 87,
            "hour": 6,
            "dayofweek": 4,
            "month": 11,
        },
    ])

    predictions = predict_temperature(model, examples)

    print("Predictions:")
    print(predictions)


if __name__ == "__main__":
    test_weather_model_inference()
