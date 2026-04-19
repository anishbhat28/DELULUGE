"""
dataset.py
==========

PyTorch Dataset for SSH forecasting.

Given the preprocessed splits.npz, this serves sliding windows:
    input:  SSH at days (t-6, t-5, ..., t)     -> shape (7, H, W)
    target: SSH at day   (t+1)                  -> shape (1, H, W)

Usage:
    from dataset import make_loaders
    train_loader, val_loader, test_loader, meta = make_loaders(batch_size=16)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os


SPLITS_PATH = "data/processed/splits.npz"

# Window config
HISTORY = 7      # days of history (channels)
HORIZON = 1      # days ahead to predict


class SSHWindowDataset(Dataset):
    """Serves (history, target) windows from a time series of SSH fields."""

    def __init__(self, arr):
        """
        arr: np.ndarray of shape (T, H, W), already normalized
        """
        self.arr = arr
        self.T, self.H, self.W = arr.shape
        # Number of valid (history, target) pairs
        self.n = self.T - HISTORY - HORIZON + 1
        assert self.n > 0, f"Need at least {HISTORY + HORIZON} timesteps, got {self.T}"

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        # History: days [i, i+1, ..., i+HISTORY-1]  -> shape (HISTORY, H, W)
        x = self.arr[i : i + HISTORY]
        # Target: day i + HISTORY + HORIZON - 1  -> shape (H, W), then (1, H, W)
        y = self.arr[i + HISTORY + HORIZON - 1][None]
        return (
            torch.from_numpy(x.copy()).float(),
            torch.from_numpy(y.copy()).float(),
        )


def make_loaders(batch_size=16, num_workers=0):
    """Build train/val/test loaders and return metadata dict."""
    assert os.path.exists(SPLITS_PATH), (
        f"{SPLITS_PATH} not found. Run preprocess.py first."
    )

    data = np.load(SPLITS_PATH)
    train_arr = data["train"]
    val_arr = data["val"]
    test_arr = data["test"]

    meta = {
        "lat": data["lat"],
        "lon": data["lon"],
        "land_mask": data["land_mask"],
        "mean": float(data["mean"]),
        "std": float(data["std"]),
        "H": train_arr.shape[1],
        "W": train_arr.shape[2],
        "history": HISTORY,
        "horizon": HORIZON,
    }

    train_ds = SSHWindowDataset(train_arr)
    val_ds = SSHWindowDataset(val_arr)
    test_ds = SSHWindowDataset(test_arr)

    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")
    print(f"Field shape: ({meta['H']}, {meta['W']})")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, meta


if __name__ == "__main__":
    # Quick smoke test
    train_loader, val_loader, test_loader, meta = make_loaders(batch_size=4)
    for x, y in train_loader:
        print(f"Batch: x {x.shape} {x.dtype}, y {y.shape} {y.dtype}")
        print(f"x range [{x.min():.2f}, {x.max():.2f}], y range [{y.min():.2f}, {y.max():.2f}]")
        break
    print("dataset.py OK")
