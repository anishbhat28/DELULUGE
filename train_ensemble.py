"""
train_ensemble.py
=================

Trains 5 ensemble members with varied (width, depth, seed) and saves
per-pixel per-timestep predictions on the test set.

After running:
    checkpoints/m0.pt ... m4.pt          (trained models)
    checkpoints/m*_history.json          (training logs)
    outputs/test_predictions.npz         (ensemble predictions + targets)

Usage:
    python train_ensemble.py
"""

import os
import numpy as np
import torch

from dataset import make_loaders
from models import build_model
from train import train_one, get_device

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)


# ----- Ensemble configs -----
# Varying width, depth, and seed to produce diverse models. Diversity is
# what makes ensemble disagreement a meaningful uncertainty signal.
ENSEMBLE_CONFIGS = [
    {"base_width": 32, "depth": 3, "lr": 3e-4, "epochs": 6, "batch_size": 16, "seed": 0},
    {"base_width": 32, "depth": 3, "lr": 3e-4, "epochs": 6, "batch_size": 16, "seed": 1},
    {"base_width": 48, "depth": 3, "lr": 2e-4, "epochs": 6, "batch_size": 16, "seed": 2},
    {"base_width": 24, "depth": 4, "lr": 3e-4, "epochs": 6, "batch_size": 16, "seed": 3},
    {"base_width": 32, "depth": 2, "lr": 4e-4, "epochs": 6, "batch_size": 16, "seed": 4},
]


def train_all():
    results = []
    for i, cfg in enumerate(ENSEMBLE_CONFIGS):
        mid = f"m{i}"
        print(f"\n{'='*60}\nTraining {mid} with {cfg}\n{'='*60}")
        r = train_one(cfg, model_id=mid)
        results.append(r)
    return results


def predict_on_test():
    """Load each trained model and run inference on the test set.
    Saves predictions, targets, and per-model errors."""
    device = get_device()
    _, _, test_loader, meta = make_loaders(batch_size=32)

    # Gather all test targets and ensemble predictions
    preds_per_model = []  # list of (T_test, H, W)
    targets = None

    for i, cfg in enumerate(ENSEMBLE_CONFIGS):
        mid = f"m{i}"
        ckpt_path = os.path.join("checkpoints", f"{mid}.pt")
        if not os.path.exists(ckpt_path):
            print(f"Skipping {mid}, no checkpoint found")
            continue

        print(f"Running inference for {mid}...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(ckpt["config"]).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        model_preds = []
        model_targets = []
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                p = model(x).cpu().numpy()[:, 0]  # (B, H, W)
                model_preds.append(p)
                if i == 0:
                    model_targets.append(y.cpu().numpy()[:, 0])

        preds = np.concatenate(model_preds, axis=0).astype(np.float32)  # (T_test, H, W)
        preds_per_model.append(preds)
        if i == 0:
            targets = np.concatenate(model_targets, axis=0).astype(np.float32)

    ensemble_preds = np.stack(preds_per_model, axis=0)  # (M, T_test, H, W)
    ensemble_mean = ensemble_preds.mean(axis=0)         # (T_test, H, W)
    ensemble_std = ensemble_preds.std(axis=0)           # disagreement field
    error = ensemble_mean - targets                     # signed error field
    abs_error = np.abs(error)

    out_path = os.path.join(OUT_DIR, "test_predictions.npz")
    np.savez_compressed(
        out_path,
        ensemble_preds=ensemble_preds,   # (M, T, H, W) - all individual preds
        ensemble_mean=ensemble_mean,     # (T, H, W)
        ensemble_std=ensemble_std,       # (T, H, W) - DISAGREEMENT FIELD
        targets=targets,                 # (T, H, W)
        error=error,                     # (T, H, W) signed
        abs_error=abs_error,             # (T, H, W)
        land_mask=meta["land_mask"],
        lat=meta["lat"],
        lon=meta["lon"],
        mean_norm=meta["mean"],          # for denormalization
        std_norm=meta["std"],
    )
    print(f"\nSaved {out_path}")
    print(f"  ensemble_preds: {ensemble_preds.shape}")
    print(f"  targets:        {targets.shape}")
    print(f"  abs_error (all ocean): mean={abs_error[:, ~meta['land_mask']].mean():.4f}")
    print(f"  disagreement (all ocean): mean={ensemble_std[:, ~meta['land_mask']].mean():.4f}")


if __name__ == "__main__":
    train_all()
    predict_on_test()
    print("\nDone. Ensemble trained and test predictions saved.")
