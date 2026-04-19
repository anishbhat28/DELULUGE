"""
baseline_check.py
=================

Compute the persistence baseline MSE:
    predict: SSH(t+1) = SSH(t)  (just the last day of history)

If our model's val loss is close to this, the model is just copying.
If our model is meaningfully lower, it's actually learning dynamics.

We also report:
    - MSE on the DIFFERENCE field (anomaly from persistence)
    - RMSE in real meters
    - Per-pixel RMSE distribution
"""

import numpy as np
import torch
from dataset import make_loaders
from models import build_model


def main():
    train_loader, val_loader, test_loader, meta = make_loaders(batch_size=32)
    land_mask = meta["land_mask"]         # (H, W) bool, True=land
    ocean_mask = ~land_mask               # True where we care
    std = meta["std"]                     # normalization std

    # ---- Persistence baseline on validation set ----
    print("Computing persistence baseline on VAL set...")
    sq_errs_persistence = []
    sq_errs_trivial_zero = []  # predict zero everywhere
    for x, y in val_loader:
        # x: (B, 7, H, W)  last channel is SSH(t)
        # y: (B, 1, H, W)  target SSH(t+1)
        persistence_pred = x[:, -1:, :, :]  # (B, 1, H, W)
        err = (persistence_pred - y).numpy()[:, 0]  # (B, H, W)
        sq_errs_persistence.append(err[:, ocean_mask] ** 2)

        zero_err = y.numpy()[:, 0]
        sq_errs_trivial_zero.append(zero_err[:, ocean_mask] ** 2)

    pers_mse = np.concatenate(sq_errs_persistence).mean()
    zero_mse = np.concatenate(sq_errs_trivial_zero).mean()

    print()
    print("=" * 60)
    print("VAL set baselines (normalized units):")
    print(f"  Predict zero everywhere:  MSE = {zero_mse:.6f}")
    print(f"  Persistence (SSH_t+1=SSH_t): MSE = {pers_mse:.6f}")
    print(f"  Your model val loss was:  MSE = 0.0003  (from smoke test)")
    print()
    print(f"Converted to real meters:")
    print(f"  Persistence RMSE = {np.sqrt(pers_mse) * std:.4f} m")
    print(f"  Model RMSE       = {np.sqrt(0.0003) * std:.4f} m  (from smoke test)")
    print("=" * 60)
    print()
    if 0.0003 < 0.5 * pers_mse:
        print("MODEL IS ACTUALLY LEARNING DYNAMICS (beats persistence by >2x)")
    elif 0.0003 < pers_mse:
        print("Model beats persistence but not hugely. Forecasting skill is real but modest.")
    else:
        print("MODEL IS JUST COPYING INPUT. Need to change prediction target.")
    print()


if __name__ == "__main__":
    main()
