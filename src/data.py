"""FER-2013 loading (Kaggle image-folder version), with a .npz cache."""
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, LABELS, SEED

FER_DIR = DATA_DIR / "fer2013"


def load_split(split: str):
    """Return (X, y): X uint8 of shape (N, 48, 48), y int64 in LABELS order."""
    cache = DATA_DIR / f"fer2013_{split}.npz"
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]

    images, targets = [], []
    for idx, name in enumerate(LABELS):
        for path in sorted((FER_DIR / split / name.lower()).glob("*.jpg")):
            images.append(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE))
            targets.append(idx)
    if not images:
        raise FileNotFoundError(f"No images found in {FER_DIR / split}")

    X, y = np.stack(images), np.array(targets, dtype=np.int64)
    np.savez_compressed(cache, X=X, y=y)
    return X, y


def train_val_split(X, y, val_size=0.1):
    """Stratified validation split taken from train (the Kaggle split has no validation set)."""
    return train_test_split(X, y, test_size=val_size, stratify=y, random_state=SEED)
