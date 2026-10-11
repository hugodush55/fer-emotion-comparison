"""Results table for the submission: mean latency per model (ms) and accuracy.

Measured with the app pipeline (same functions, same timing) on FER-2013 test images,
which are already cropped to 48x48. First calls are excluded (warm-up in EmotionPipeline).
Haar detection is measured separately, on 640x480 images containing one face.

    python -m src.benchmark
"""
import json
import time

import cv2
import numpy as np
import torch

from src.config import REPORTS_DIR, SEED
from src.data import load_split
from src.pipeline import MODEL_NAMES, EmotionPipeline

N_LATENCY = 500
N_ACC_SMALL = 50


def run(pipeline, X, y):
    lat = {k: [] for k in MODEL_NAMES}
    correct = {k: [] for k in MODEL_NAMES}
    for img, target in zip(X, y):
        for pred in pipeline.predict_face(img):
            lat[pred.model].append(pred.latency_ms)
            correct[pred.model].append(int(pred.probs.argmax()) == int(target))
    return lat, correct


def detection_latency(pipeline, X, n=200):
    """FER face upscaled and pasted into a 640x480 image: detection + crop."""
    rng = np.random.default_rng(SEED)
    times, found = [], 0
    for img in X[:n]:
        canvas = np.full((480, 640), int(rng.integers(60, 200)), np.uint8)
        size = int(rng.integers(160, 300))
        x, y = int(rng.integers(0, 640 - size)), int(rng.integers(0, 480 - size))
        canvas[y:y + size, x:x + size] = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        res = pipeline.process(canvas)
        times.append(res.detect_ms)
        found += res.face48 is not None
    return float(np.mean(times)), found / n


def main():
    X, y = load_split("test")
    rng = np.random.default_rng(SEED)
    idx_lat = rng.choice(len(X), N_LATENCY, replace=False)

    cpu = EmotionPipeline(device="cpu")
    lat, correct = run(cpu, X[idx_lat], y[idx_lat])
    det_ms, det_rate = detection_latency(cpu, X[idx_lat])
    gpu_ms = None
    if torch.cuda.is_available():
        gpu = EmotionPipeline(device="cuda")
        times = []
        for img in X[idx_lat]:
            t0 = time.perf_counter()
            gpu._cnn(img)
            times.append((time.perf_counter() - t0) * 1000)
        gpu_ms = float(np.mean(times))

    rows = []
    for name, label in MODEL_NAMES.items():
        metrics = json.loads((REPORTS_DIR / f"{name}_metrics.json").read_text())
        rows.append({
            "model": label,
            "latency_ms_mean": float(np.mean(lat[name])),
            "latency_ms_std": float(np.std(lat[name])),
            "latency_ms_p95": float(np.percentile(lat[name], 95)),
            f"accuracy_{N_ACC_SMALL}": float(np.mean(correct[name][:N_ACC_SMALL])),
            f"accuracy_{N_LATENCY}": float(np.mean(correct[name])),
            "accuracy_full_test": metrics["test_accuracy"],
        })

    results = {"n_latency": N_LATENCY, "detection_ms_mean": det_ms, "detection_rate": det_rate,
               "cnn_gpu_latency_ms_mean": gpu_ms, "models": rows}
    (REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2))

    lines = [
        f"| Model | Mean latency (ms) | Std / p95 (ms) | Accuracy {N_ACC_SMALL} images "
        f"| Accuracy {N_LATENCY} images | Accuracy full test set (7178) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['model']} | **{r['latency_ms_mean']:.1f}** | {r['latency_ms_std']:.1f} / "
                     f"{r['latency_ms_p95']:.1f} | {r[f'accuracy_{N_ACC_SMALL}']:.0%} | "
                     f"{r[f'accuracy_{N_LATENCY}']:.1%} | **{r['accuracy_full_test']:.1%}** |")
    notes = [
        "",
        f"Latencies on CPU, {N_LATENCY} test images, first calls excluded; "
        "latency = feature extraction + inference (+ flip TTA for the CNN).",
        f"Haar detection + crop (640x480 image): {det_ms:.1f} ms on average, shared by the 3 models "
        f"(face found in {det_rate:.0%} of the synthetic test images).",
    ]
    if gpu_ms:
        notes.append(f"mini-Xception on GPU ({torch.cuda.get_device_name(0)}): {gpu_ms:.1f} ms.")
    table = "\n".join(lines + notes)
    (REPORTS_DIR / "results.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
