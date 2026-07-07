from pathlib import Path
import os

# Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model Directory
MODEL_DIR = PROJECT_ROOT / "models"

# Reports Directory
REPORT_DIR = PROJECT_ROOT / "reports"

# Application Directory
APP_DIR = PROJECT_ROOT / "app"

# Ensure required directories exist (useful for fresh checkouts / production)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


# Default File Paths
RAW_DATA_PATH = RAW_DATA_DIR / "uber.csv"

CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "uber_clean.csv"

FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "uber_feature_engineered.csv"

MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"

BEST_MODEL_PATH = MODEL_DIR / "best_random_forest.pkl"

RANDOM_STATE = 42

TEST_SIZE = 0.20

MAX_PASSENGERS = 6

EARTH_RADIUS_KM = 6371

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000",
)
PREDICTION_API_URL = f"{API_BASE_URL}/predict"