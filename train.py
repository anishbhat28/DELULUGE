"""
train.py
========

Train a single SSH forecasting model.

Key choices:
- Loss is masked so land pixels don't contribute (prediction on land is meaningless)
- Uses MPS (Apple Silicon) if available, else CUDA, else CPU
- Saves best-val checkpoint and training history

Usage:
    from train import train_one
    history = train_one(config, model_id="m0")
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from dataset import make_loaders
from models import build_model, count_params


CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def masked_mse(pred, target, mask):
    """
    MSE loss that ignores land pixels.
    pred, target: (B, 1, H, W)
    mask: (H, W) bool, True = land (ignore), False = ocean (keep)
    """
    ocean = ~mask  # True where we care
    ocean = ocean.to(pred.device).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    sq_err = (pred - target) ** 2
    # Mean over ocean pixels only
    return (sq_err * ocean).sum() / ocean.sum() / pred.shape[0]


def train_one(config, model_id="m0", verbose=True):
    """
    Train one model. Returns dict with best_val_loss and history.

    config keys:
        base_width: int
        depth: int
        lr: float
        epochs: int
        batch_size: int
        seed: int
    """
    torch.manual_seed(config.get("seed", 0))
    np.random.seed(config.get("seed", 0))

    device = get_device()
    if verbose:
        print(f"[{model_id}] Device: {device}")

    # Data
    train_loader, val_loader, _, meta = make_loaders(
        batch_size=config.get("batch_size", 16),
    )
    land_mask = torch.from_numpy(meta["land_mask"])  # (H, W) bool

    # Model
    model = build_model(config).to(device)
    n_params = count_params(model)
    if verbose:
        print(f"[{model_id}] Model: base_width={config['base_width']}, "
              f"depth={config['depth']}, params={n_params:,}")

    # Optim
    optimizer = AdamW(model.parameters(), lr=config.get("lr", 3e-4), weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.get("epochs", 8))

    best_val = float("inf")
    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    ckpt_path = os.path.join(CHECKPOINT_DIR, f"{model_id}.pt")

    for epoch in range(config.get("epochs", 8)):
        t0 = time.time()

        # --- Train ---
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = masked_mse(pred, y, land_mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()

        # --- Val ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = masked_mse(pred, y, land_mask)
                val_losses.append(loss.item())

        tr = float(np.mean(train_losses))
        vl = float(np.mean(val_losses))
        dt = time.time() - t0
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        history["epoch_time"].append(dt)

        if verbose:
            print(f"[{model_id}] epoch {epoch+1}/{config['epochs']} "
                  f"train={tr:.4f} val={vl:.4f} ({dt:.1f}s)")

        if vl < best_val:
            best_val = vl
            torch.save({
                "state_dict": model.state_dict(),
                "config": config,
                "meta": {k: v for k, v in meta.items() if not isinstance(v, np.ndarray)},
                "epoch": epoch,
                "val_loss": vl,
            }, ckpt_path)

    # Save history
    hist_path = os.path.join(CHECKPOINT_DIR, f"{model_id}_history.json")
    with open(hist_path, "w") as f:
        json.dump({
            "config": config,
            "best_val": best_val,
            "n_params": n_params,
            "history": history,
        }, f, indent=2)

    if verbose:
        print(f"[{model_id}] Best val loss: {best_val:.4f}, saved to {ckpt_path}")
    return {"best_val": best_val, "history": history, "ckpt_path": ckpt_path}


if __name__ == "__main__":
    # Single-model smoke test
    config = {
        "base_width": 32,
        "depth": 3,
        "lr": 3e-4,
        "epochs": 3,
        "batch_size": 16,
        "seed": 0,
    }
    train_one(config, model_id="smoke_test")
