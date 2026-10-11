"""Model 3: mini-Xception training on FER-2013.

Adam (lr 1e-3, warmup then cosine decay), batch 64, label smoothing,
GPU augmentations, horizontal-flip TTA at inference.

Two modes:
  --mode val   trains on train (90 %) and tracks validation (10 %): used to pick
               the width and to check that the schedule does not overfit.
  --mode full  retrains on the whole train set with the same schedule, saves
               models/mini_xception.pt and evaluates on test.

    python -m src.train_cnn --mode val --width 2
    python -m src.train_cnn --mode full --width 2

Regularization options: --erasing (random erasing) and --mix-prob (mixup/cutmix).
"""
import argparse
import json
import math
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from src.cnn import MiniXception, augment, mix_batch, predict_proba, random_erasing
from src.config import MODELS_DIR, REPORTS_DIR, SEED
from src.data import load_split, train_val_split
from src.evaluation import evaluate

BATCH_SIZE = 64
WARMUP_EPOCHS = 3


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["val", "full"], default="val")
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--erasing", type=float, default=0.0, help="random erasing probability")
    ap.add_argument("--mix-prob", type=float, default=0.0, help="per-batch mixup/cutmix probability")
    ap.add_argument("--tag", default="", help="report file suffix")
    return ap.parse_args()


def to_tensor(X, device):
    return torch.from_numpy(X).float().div_(255).unsqueeze(1).to(device)


def batched_proba(model, X, tta, batch_size=1024):
    return torch.cat([predict_proba(model, X[i:i + batch_size], tta) for i in range(0, len(X), batch_size)])


def warmup_cosine(epochs):
    """LR factor: linear warmup over WARMUP_EPOCHS, then cosine decay to 0."""
    def factor(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / max(1, epochs - WARMUP_EPOCHS)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return factor


def plot_history(history, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax1.plot(epochs, history["train_loss"], label="train")
    ax2.plot(epochs, history["train_acc"], label="train")
    if history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], label="val")
        ax2.plot(epochs, history["val_acc"], label="val (TTA)")
    ax1.set(title="Loss (with label smoothing)", xlabel="epoch")
    ax2.set(title="Accuracy", xlabel="epoch")
    for ax in (ax1, ax2):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = f"mini_xception_w{args.width}_{args.mode}" + (f"_{args.tag}" if args.tag else "")

    X_train, y_train = load_split("train")
    X_test, y_test = load_split("test")
    if args.mode == "val":
        X_fit, X_val, y_fit, y_val = train_val_split(X_train, y_train)
        X_val, y_val = to_tensor(X_val, device), torch.from_numpy(y_val).to(device)
    else:
        X_fit, y_fit = X_train, y_train
    X_fit, y_fit = to_tensor(X_fit, device), torch.from_numpy(y_fit).to(device)
    X_te = to_tensor(X_test, device)
    print(f"{tag} | device {device} | fit {len(X_fit)} | test {len(X_test)}")

    model = MiniXception(width=args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,}")
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(args.epochs))

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "lr")}
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(X_fit), device=device)
        total_loss, correct = 0.0, 0
        for i in range(0, len(perm), BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            xb, yb = augment(X_fit[idx]), y_fit[idx]
            if args.erasing > 0:
                xb = random_erasing(xb, p=args.erasing)
            if args.mix_prob > 0 and torch.rand(1).item() < args.mix_prob:
                xb, yb, yb2, lam = mix_batch(xb, yb)
                logits = model(xb)
                loss = lam * criterion(logits, yb) + (1 - lam) * criterion(logits, yb2)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
        history["lr"].append(optimizer.param_groups[0]["lr"])
        scheduler.step()
        history["train_loss"].append(total_loss / len(X_fit))
        history["train_acc"].append(correct / len(X_fit))

        line = (f"epoch {epoch:3d} | train loss {history['train_loss'][-1]:.4f} "
                f"acc {history['train_acc'][-1]:.4f} | lr {history['lr'][-1]:.1e}")
        if args.mode == "val":
            p_val = batched_proba(model, X_val, tta=True)
            history["val_loss"].append(nn.functional.nll_loss(p_val.log(), y_val).item())
            history["val_acc"].append((p_val.argmax(1) == y_val).float().mean().item())
            line += f" | val loss {history['val_loss'][-1]:.4f} acc {history['val_acc'][-1]:.4f}"
        print(line, flush=True)

    train_time = time.perf_counter() - t_start
    REPORTS_DIR.mkdir(exist_ok=True)
    plot_history(history, REPORTS_DIR / f"{tag}_curves.png")
    (REPORTS_DIR / f"{tag}_history.json").write_text(json.dumps(history, indent=2))

    acc_no_tta = (batched_proba(model, X_te, tta=False).argmax(1).cpu().numpy() == y_test).mean()
    y_pred = batched_proba(model, X_te, tta=True).argmax(1).cpu().numpy()
    summary = {
        "width": args.width, "params": n_params, "train_time_s": round(train_time, 1),
        "test_accuracy_no_tta": float(acc_no_tta),
        "hyperparams": {"batch_size": BATCH_SIZE, "epochs": args.epochs, "lr": args.lr,
                        "schedule": f"warmup {WARMUP_EPOCHS} + cosine", "weight_decay": args.weight_decay,
                        "label_smoothing": args.label_smoothing, "random_erasing": args.erasing,
                        "mixup_cutmix_prob": args.mix_prob},
    }
    if args.mode == "val":
        summary["final_val_accuracy"] = history["val_acc"][-1]
        summary["best_val_accuracy"] = max(history["val_acc"])
        summary["test_accuracy_tta"] = float((y_pred == y_test).mean())
        (REPORTS_DIR / f"{tag}_metrics.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
    else:
        MODELS_DIR.mkdir(exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "width": args.width, "tta": True},
                   MODELS_DIR / "mini_xception.pt")
        evaluate("mini_xception", y_test, y_pred, extra=summary)
        print(f"without TTA: {acc_no_tta:.4f}")
    print(f"Training time: {train_time / 60:.1f} min")


if __name__ == "__main__":
    main()
