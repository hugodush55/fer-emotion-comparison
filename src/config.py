"""Constantes partagées par les notebooks, l'entraînement et l'application."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

IMG_SIZE = 48

# Ordre officiel FER-2013 : à respecter partout (datasets, modèles, interface).
LABELS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
LABEL_TO_IDX = {name.lower(): i for i, name in enumerate(LABELS)}

SEED = 42
