"""Métriques communes aux 3 modèles : accuracy, rapport par classe, matrice de confusion."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from src.config import LABELS, REPORTS_DIR


def evaluate(name, y_true, y_pred, extra=None):
    """Affiche et sauvegarde les métriques dans reports/<name>_*."""
    REPORTS_DIR.mkdir(exist_ok=True)
    acc = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=LABELS, digits=3)
    print(f"\n[{name}] accuracy test = {acc:.4f}\n{report}")

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", values_format=".2f", colorbar=False)
    ax.set_title(f"{name} : accuracy {acc:.1%} (normalisée par ligne)")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / f"{name}_confusion.png", dpi=150)
    plt.close(fig)

    metrics = {"model": name, "test_accuracy": acc, **(extra or {})}
    (REPORTS_DIR / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    (REPORTS_DIR / f"{name}_report.txt").write_text(report)
    return acc
