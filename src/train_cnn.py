"""Modèle 3 : entraînement de mini-Xception sur FER-2013.

Adam (lr 1e-3) + ReduceLROnPlateau, batch 64, early stopping sur la val-loss,
augmentations sur GPU. Le meilleur modèle (val-loss minimale) est évalué sur test.

    python -m src.train_cnn
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.cnn import MiniXception, augment
from src.config import MODELS_DIR, REPORTS_DIR, SEED
from src.data import load_split, train_val_split
from src.evaluation import evaluate

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 150
LR_PATIENCE = 6
EARLY_STOP_PATIENCE = 20


def to_tensor(X, device):
    return torch.from_numpy(X).float().div_(255).unsqueeze(1).to(device)


@torch.no_grad()
def predict(model, X, batch_size=1024):
    model.eval()
    logits = torch.cat([model(X[i:i + batch_size]) for i in range(0, len(X), batch_size)])
    return logits


def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"], label="val")
    ax1.set(title="Loss", xlabel="epoch")
    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"], label="val")
    ax2.set(title="Accuracy", xlabel="epoch")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "mini_xception_curves.png", dpi=150)
    plt.close(fig)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    X_tr, X_val, y_tr, y_val = train_val_split(X_train, y_train)
    print(f"device {device} | train {len(X_tr)} | val {len(X_val)} | test {len(X_test)}")

    X_tr, X_val, X_te = (to_tensor(a, device) for a in (X_tr, X_val, X_test))
    y_tr, y_val = (torch.from_numpy(a).to(device) for a in (y_tr, y_val))

    model = MiniXception().to(device)
    print(f"paramètres : {sum(p.numel() for p in model.parameters()):,}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=LR_PATIENCE)

    MODELS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    weights_path = MODELS_DIR / "mini_xception.pt"
    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr")}
    best_val_loss, best_epoch, wait = float("inf"), 0, 0
    t_start = time.perf_counter()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        total_loss, correct = 0.0, 0
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = augment(X_tr[idx]), y_tr[idx]
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()

        val_logits = predict(model, X_val)
        val_loss = criterion(val_logits, y_val).item()
        val_acc = (val_logits.argmax(1) == y_val).float().mean().item()
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        for k, v in zip(history, (total_loss / len(X_tr), correct / len(X_tr), val_loss, val_acc, lr)):
            history[k].append(v)
        flag = ""
        if val_loss < best_val_loss:
            best_val_loss, best_epoch, wait = val_loss, epoch, 0
            torch.save(model.state_dict(), weights_path)
            flag = " *"
        else:
            wait += 1
        print(f"epoch {epoch:3d} | train loss {history['train_loss'][-1]:.4f} acc {history['train_acc'][-1]:.4f} "
              f"| val loss {val_loss:.4f} acc {val_acc:.4f} | lr {lr:.1e}{flag}", flush=True)
        if wait >= EARLY_STOP_PATIENCE:
            print(f"Early stopping (meilleure epoch : {best_epoch})")
            break

    train_time = time.perf_counter() - t_start
    plot_history(history)
    (REPORTS_DIR / "mini_xception_history.json").write_text(json.dumps(history, indent=2))

    model.load_state_dict(torch.load(weights_path, map_location=device))
    y_pred = predict(model, X_te).argmax(1).cpu().numpy()
    best_val_acc = history["val_acc"][best_epoch - 1]
    evaluate("mini_xception", y_test, y_pred, extra={
        "best_epoch": best_epoch, "best_val_loss": best_val_loss, "val_accuracy": best_val_acc,
        "train_time_s": round(train_time, 1),
        "hyperparams": {"batch_size": BATCH_SIZE, "lr": LR, "weight_decay": WEIGHT_DECAY,
                        "lr_patience": LR_PATIENCE, "early_stop_patience": EARLY_STOP_PATIENCE},
    })
    print(f"Entraînement : {train_time / 60:.1f} min | poids : {weights_path}")


if __name__ == "__main__":
    main()
