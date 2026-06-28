from pathlib import Path

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

# Default File Paths
RAW_DATA_PATH = RAW_DATA_DIR / "uber.csv"

CLEAN_DATA_PATH = PROCESSED_DATA_DIR / "uber_clean.csv"

FEATURE_DATA_PATH = PROCESSED_DATA_DIR / "uber_feature_engineered.csv"

MODEL_PATH = MODEL_DIR / "random_forest_model.pkl"

BEST_MODEL_PATH = MODEL_DIR / "best_random_forest.pkl"