"""Metrics shared by the 3 models: accuracy, per-class report, confusion matrix."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.config import LABELS, REPORTS_DIR


def evaluate(name, y_true, y_pred, extra=None):
    """Print and save the metrics to reports/<name>_*."""
    REPORTS_DIR.mkdir(exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=LABELS, digits=3)
    print(f"\n[{name}] test accuracy = {acc:.4f}\n{report}")

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(f"{name}: accuracy {acc:.1%} (row-normalized)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"{name}_confusion.png", dpi=150)
    plt.close(fig)

    np.savez_compressed(REPORTS_DIR / f"{name}_predictions.npz", y_true=y_true, y_pred=y_pred)
    metrics = {"model": name, "test_accuracy": acc, **(extra or {})}
    (REPORTS_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS_DIR / f"{name}_report.txt").write_text(report)
    return acc


def plot_all_confusions(names, titles, path):
    """Confusion matrices of several models side by side (read back from their saved predictions)."""
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5.5))
    for ax, name, title in zip(axes, names, titles):
        d = np.load(REPORTS_DIR / f"{name}_predictions.npz")
        cm = confusion_matrix(d["y_true"], d["y_pred"], normalize="true")
        acc = accuracy_score(d["y_true"], d["y_pred"])
        ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
        ax.set_title(f"{title}: {acc:.1%}")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
