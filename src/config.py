"""Constants shared by training, evaluation and the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

IMG_SIZE = 48

# Official FER-2013 label order: must be the same everywhere (datasets, models, UI).
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
LABEL_TO_IDX = {name.lower(): i for i, name in enumerate(LABELS)}

SEED = 42
