from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "weather.db"

RANDOM_STATE = 42

LATITUDE = 42.7022
LONGITUDE = 9.4512

DEFAULT_START = "2023-01-01"
DEFAULT_END   = "2024-12-31"


