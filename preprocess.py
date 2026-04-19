"""
preprocess.py
=============

Loads the raw Scripps Gulf of Mexico SSH .nc file, extracts the Loop Current
subdomain, handles land mask (NaN), normalizes, and saves train/val/test
splits as a single compressed .npz file.

Usage:
    python preprocess.py

Output:
    data/processed/splits.npz
        contains: train, val, test arrays, each of shape (T, H, W)
        plus: lat, lon, land_mask, mean, std (normalization stats)
"""

import numpy as np
import xarray as xr
import os

# ---------- Config ----------
RAW_PATH = os.path.expanduser("~/Downloads/run2_clim_v2_ssh.nc")

# Loop Current subdomain defined by real coordinates (not array indices).
# The Loop Current sits roughly 22-28 N and 92-84 W in the eastern Gulf.
LAT_MIN, LAT_MAX = 22.0, 28.0
LON_MIN, LON_MAX = -92.0, -84.0

# Time splits (14373 days total)
#   train: days 0     ..  9999  (~27.4 years)
#   val:   days 10000 .. 11500  (~4.1 years)
#   test:  days 11500 .. 14373  (~7.9 years)
TRAIN_END = 10000
VAL_END = 11500

OUT_DIR = "data/processed"
OUT_PATH = os.path.join(OUT_DIR, "splits.npz")


def main():
    print(f"Loading {RAW_PATH} ...")
    ds = xr.open_dataset(RAW_PATH, decode_times=False)

    # Extract subdomain using real coordinates (no index guessing)
    ssh = ds["ssh"].sel(
        lat=slice(LAT_MIN, LAT_MAX),
        lon=slice(LON_MIN, LON_MAX),
    )
    lat = ssh["lat"].values
    lon = ssh["lon"].values

    print(f"Subdomain: lat {lat.min():.2f} to {lat.max():.2f}, "
          f"lon {lon.min():.2f} to {lon.max():.2f}")
    print(f"Shape: {ssh.shape} (time, lat, lon)")

    # Force load (this reads from disk — ~275 MB into memory)
    print("Loading into memory (this may take 30-60 seconds)...")
    ssh_arr = ssh.values.astype(np.float32)
    print(f"Loaded. Array shape: {ssh_arr.shape}, dtype: {ssh_arr.dtype}")

    # Build land mask from first timestep
    # A pixel is land if it's NaN at any timestep; we use t=0 as a proxy
    # (if land, it's land throughout).
    land_mask = np.isnan(ssh_arr[0])
    print(f"Land pixels: {land_mask.sum()} / {land_mask.size} "
          f"({100*land_mask.mean():.1f}%)")

    # Replace NaN (land) with 0. We'll also save the mask so models can
    # ignore land pixels if they want.
    ssh_arr = np.nan_to_num(ssh_arr, nan=0.0)

    # Normalize using TRAIN-ONLY statistics (no test leakage)
    train_vals = ssh_arr[:TRAIN_END][:, ~land_mask]  # ocean pixels only
    mean = float(train_vals.mean())
    std = float(train_vals.std())
    print(f"Train stats (ocean only): mean={mean:.4f} m, std={std:.4f} m")

    # Apply normalization to everything (land is 0, stays ~0 after norm)
    ssh_norm = (ssh_arr - mean) / std
    # Re-zero land so it's exactly 0 (not -mean/std)
    ssh_norm[:, land_mask] = 0.0

    # Split
    train = ssh_norm[:TRAIN_END]
    val = ssh_norm[TRAIN_END:VAL_END]
    test = ssh_norm[VAL_END:]
    print(f"Splits: train {train.shape}, val {val.shape}, test {test.shape}")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        OUT_PATH,
        train=train,
        val=val,
        test=test,
        lat=lat,
        lon=lon,
        land_mask=land_mask,
        mean=mean,
        std=std,
    )
    size_mb = os.path.getsize(OUT_PATH) / 1e6
    print(f"Saved {OUT_PATH} ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()
