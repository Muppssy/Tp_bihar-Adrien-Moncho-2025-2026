from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "weather.db"

RANDOM_STATE = 42

LATITUDE = 42.7022
LONGITUDE = 9.4512

DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE   = "2024-12-31"


BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "weather.db"
MODEL_PATH = BASE_DIR / "model" / "registry" / "weather_rf_v1.joblib"
METADATA_PATH = BASE_DIR / "model" / "registry" / "weather_rf_v1_metadata.json"
