"""Model 1: LBP (4x4 blocks) + KNN.

Hyperparameters are selected on a validation split taken from train, then the model
is refit on the whole train set and evaluated on test.

    python -m src.train_knn
"""
import itertools
import json
import time

import joblib
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from src.config import MODELS_DIR, REPORTS_DIR
from src.data import load_split, train_val_split
from src.evaluation import evaluate
from src.features import batch_features, lbp_features

LBP_GRID = [(8, 1), (8, 2), (8, 3), (16, 2)]  # (P, R)
K_GRID = [1, 3, 5, 9, 15, 25, 35, 50]
WEIGHTS_GRID = ["uniform", "distance"]
P_METRIC_GRID = [1, 2]  # 1 = Manhattan, 2 = Euclidean


def main():
    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train)
    print(f"train {len(X_tr)} | val {len(X_val)} | test {len(X_test)}")

    results, best = [], None
    for P, R in LBP_GRID:
        t0 = time.perf_counter()
        F_tr = batch_features(lbp_features, X_tr, P=P, R=R)
        F_val = batch_features(lbp_features, X_val, P=P, R=R)
        print(f"\nLBP P={P} R={R}: {F_tr.shape[1]} features ({time.perf_counter() - t0:.1f}s)")
        for k, weights, p in itertools.product(K_GRID, WEIGHTS_GRID, P_METRIC_GRID):
            knn = KNeighborsClassifier(n_neighbors=k, weights=weights, p=p, n_jobs=-1).fit(F_tr, y_tr)
            acc = accuracy_score(y_val, knn.predict(F_val))
            row = {"P": P, "R": R, "k": k, "weights": weights, "p": p, "val_accuracy": acc}
            results.append(row)
            print(f"  k={k:<3} {weights:<8} p={p} -> val {acc:.4f}")
            if best is None or acc > best["val_accuracy"]:
                best = row

    print(f"\nBest config: {best}")
    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "lbp_knn_grid.json").write_text(json.dumps(results, indent=2))

    lbp_params = {"P": best["P"], "R": best["R"], "method": "uniform", "grid": 4}
    F_train = batch_features(lbp_features, X_train, **lbp_params)
    F_test = batch_features(lbp_features, X_test, **lbp_params)
    knn = KNeighborsClassifier(n_neighbors=best["k"], weights=best["weights"], p=best["p"], n_jobs=-1)
    knn.fit(F_train, y_train)
    evaluate("lbp_knn", y_test, knn.predict(F_test), extra={"best_params": best, "lbp_params": lbp_params})

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": knn, "lbp_params": lbp_params}, MODELS_DIR / "lbp_knn.joblib", compress=3)
    print(f"Model saved to {MODELS_DIR / 'lbp_knn.joblib'}")


if __name__ == "__main__":
    main()
