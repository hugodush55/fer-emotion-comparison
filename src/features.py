"""Extraction de features classiques (LBP, HOG) sur des visages 48x48 en niveaux de gris."""
import numpy as np
from joblib import Parallel, delayed
from skimage.feature import local_binary_pattern


def lbp_features(img, P=8, R=1, method="uniform", grid=4):
    """LBP découpé en grid x grid blocs : un histogramme normalisé par bloc, concaténés."""
    codes = local_binary_pattern(img, P, R, method=method)
    n_bins = P + 2 if method == "uniform" else 2 ** P
    h, w = codes.shape
    bh, bw = h // grid, w // grid
    hists = []
    for i in range(grid):
        for j in range(grid):
            block = codes[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            hist, _ = np.histogram(block, bins=n_bins, range=(0, n_bins))
            hists.append(hist / block.size)
    return np.concatenate(hists).astype(np.float32)


def batch_features(fn, X, n_jobs=-1, **params):
    """Applique fn(img, **params) à toutes les images en parallèle."""
    feats = Parallel(n_jobs=n_jobs, batch_size=512)(delayed(fn)(img, **params) for img in X)
    return np.stack(feats)
