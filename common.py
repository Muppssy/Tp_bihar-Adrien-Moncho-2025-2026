import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Reproductibilite
RANDOM_STATE = 42

# Dataset multimodal / image
TRAIN_CSV_PATH = BASE_DIR / "train.csv"
IMAGE_DIR = Path(r"C:\Users\mupps\Desktop\COMP5329S1A2Dataset\data")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Embeddings texte
GLOVE_DIR = Path(os.getenv("GLOVE_DIR", BASE_DIR / "GloVe"))
GLOVE_PATH = Path(os.getenv("GLOVE_PATH", GLOVE_DIR / "glove.6B.50d.txt"))
GLOVE_EMBEDDING_DIM = 50

# Model registry
REGISTRY_DIR = BASE_DIR / "model" / "registry"
IMAGE_MODEL_PATH = REGISTRY_DIR / "image_model.keras"
TEXT_MODEL_PATH = REGISTRY_DIR / "text_model.keras"
MULTIMODAL_MODEL_PATH = REGISTRY_DIR / "multimodal_model.keras"

# Weather / time series
DB_PATH = BASE_DIR / "data" / "weather.db"
MODEL_PATH = REGISTRY_DIR / "weather_rf_v1.joblib"
METADATA_PATH = REGISTRY_DIR / "weather_rf_v1_metadata.json"

LATITUDE = 42.7022
LONGITUDE = 9.4512

DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2024-12-31"

APP_VERSION = "0.1.0"
