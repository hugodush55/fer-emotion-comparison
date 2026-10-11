"""Model 2: HOG + linear SVM (One-vs-Rest).

C (and the HOG cell size) is selected on a validation split taken from train, then the
model is refit on the whole train set with score calibration (LinearSVC does not output
probabilities) and evaluated on test.

    python -m src.train_svm
"""
import itertools
import json
import time

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from src.config import MODELS_DIR, REPORTS_DIR, SEED
from src.data import load_split, train_val_split
from src.evaluation import evaluate
from src.features import batch_features, hog_features

CELL_GRID = [8]  # pixels per cell, as specified (6x6 was tested: +0.1 pt on validation, not kept)
C_GRID = [0.0001, 0.001, 0.01, 0.1, 1.0]


def make_svm(C):
    # LinearSVC is One-vs-Rest; dual=False because n_samples > n_features.
    return LinearSVC(C=C, dual=False, max_iter=5000, random_state=SEED)


def main():
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train)
    print(f"train {len(X_tr)} | val {len(X_val)} | test {len(X_test)}")

    results, best = [], None
    for cell in CELL_GRID:
        F_tr = batch_features(hog_features, X_tr, cell=cell)
        F_val = batch_features(hog_features, X_val, cell=cell)
        print(f"\nHOG cell={cell}: {F_tr.shape[1]} features")
        for C in C_GRID:
            t0 = time.perf_counter()
            clf = make_pipeline(StandardScaler(), make_svm(C)).fit(F_tr, y_tr)
            acc = accuracy_score(y_val, clf.predict(F_val))
            row = {"cell": cell, "C": C, "val_accuracy": acc}
            results.append(row)
            print(f"  C={C:<7} -> val {acc:.4f} ({time.perf_counter() - t0:.1f}s)")
            if best is None or acc > best["val_accuracy"]:
                best = row

    print(f"\nBest config: {best}")
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "hog_svm_grid.json").write_text(json.dumps(results, indent=2))

    hog_params = {"orientations": 9, "cell": best["cell"], "block": 2}
    F_train = batch_features(hog_features, X_train, **hog_params)
    F_test = batch_features(hog_features, X_test, **hog_params)
    clf = make_pipeline(StandardScaler(),
                        CalibratedClassifierCV(make_svm(best["C"]), method="sigmoid", cv=3))
    clf.fit(F_train, y_train)
    evaluate("hog_svm", y_test, clf.predict(F_test), extra={"best_params": best, "hog_params": hog_params})

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": clf, "hog_params": hog_params}, MODELS_DIR / "hog_svm.joblib", compress=3)
    print(f"Model saved to {MODELS_DIR / 'hog_svm.joblib'}")


if __name__ == "__main__":
    main()
