from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "reviews.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_SAVE_PATH = MODEL_DIR / "stacking_ensemble.pkl"
FEATURES_SAVE_PATH = MODEL_DIR / "feature_pipeline.pkl"

# Dataset Column Definitions
TEXT_COL = "text_"
LABEL_COL = "label"
CAT_COL = "category"