"""
regimes.py
==========

Compute physical regime indicators from the SSH field itself. These are
the 'language' in which the autoresearch loop proposes hypotheses, and
they define the colors/bins for the disagreement-vs-error scatter.

All indicators are derived purely from SSH(x, y, t) — no external data
needed. This is standard physical oceanography practice.

Indicators:
-----------
1. Eddy Kinetic Energy (EKE) proxy from SSH gradients
   - High EKE = energetic region (likely eddy-dominated)
   - Spatial field, (T, H, W)

2. Okubo-Weiss parameter from SSH curvature
   - W < 0: vortex-dominated (eddy core)
   - W > 0: strain-dominated (between eddies, fronts)
   - Spatial field, (T, H, W)

3. Loop Current northward extent
   - Scalar per timestep — how far north the Loop Current pushes
   - Measured as northernmost latitude where a specific SSH contour sits
   - Time series, (T,)

4. Domain-mean SSH anomaly magnitude
   - Scalar per timestep — overall 'activity' in the basin
   - Time series, (T,)
"""

import numpy as np


# Approximate dx, dy in meters for 0.05 degree grid at ~25N latitude
# 1 degree lat ~ 111 km; 1 degree lon at 25N ~ 100 km
DX_M = 0.05 * 100_000   # ~5 km
DY_M = 0.05 * 111_000   # ~5.5 km


def denormalize(ssh_norm, mean, std):
    """Convert normalized SSH back to meters."""
    return ssh_norm * std + mean


def eddy_kinetic_energy_proxy(ssh_m):
    """
    EKE proxy from geostrophic velocity derived from SSH gradients.

    Geostrophic: u ~ -dSSH/dy, v ~ dSSH/dx  (times g/f, constants that we
    absorb since we only care about relative EKE).

    Input:  ssh_m (T, H, W) in meters
    Output: eke   (T, H, W) relative units (proportional to |grad SSH|^2)
    """
    # Compute gradients along lat (axis 1) and lon (axis 2)
    dSSH_dy = np.gradient(ssh_m, DY_M, axis=1)
    dSSH_dx = np.gradient(ssh_m, DX_M, axis=2)
    # EKE ~ u^2 + v^2 ~ (dSSH/dy)^2 + (dSSH/dx)^2
    eke = dSSH_dy ** 2 + dSSH_dx ** 2
    return eke.astype(np.float32)


def okubo_weiss(ssh_m):
    """
    Okubo-Weiss parameter from SSH.

    Geostrophic velocity: u = -dSSH/dy, v = dSSH/dx
    Strain:    S_n = du/dx - dv/dy   (normal)
               S_s = dv/dx + du/dy   (shear)
    Vorticity: W   = dv/dx - du/dy
    Okubo-Weiss: OW = S_n^2 + S_s^2 - W^2
        OW < 0: rotation-dominated (eddy core)
        OW > 0: strain-dominated (between eddies, fronts)

    Input:  ssh_m (T, H, W)
    Output: ow    (T, H, W)
    """
    dSSH_dy = np.gradient(ssh_m, DY_M, axis=1)
    dSSH_dx = np.gradient(ssh_m, DX_M, axis=2)
    u = -dSSH_dy
    v = dSSH_dx

    du_dx = np.gradient(u, DX_M, axis=2)
    du_dy = np.gradient(u, DY_M, axis=1)
    dv_dx = np.gradient(v, DX_M, axis=2)
    dv_dy = np.gradient(v, DY_M, axis=1)

    S_n = du_dx - dv_dy
    S_s = dv_dx + du_dy
    W = dv_dx - du_dy

    ow = S_n ** 2 + S_s ** 2 - W ** 2
    return ow.astype(np.float32)


def loop_current_northward_extent(ssh_m, lat, contour_threshold_m=0.2):
    """
    Crude Loop Current northward extent: northernmost latitude where any
    longitude in the eastern half of the domain has SSH >= threshold
    relative to the per-timestep mean.

    Input:  ssh_m (T, H, W), lat (H,)
    Output: extent (T,) in degrees north
    """
    T, H, W = ssh_m.shape
    # Restrict to eastern half of domain (where Loop Current lives)
    east_half = ssh_m[:, :, W // 2:]  # (T, H, W/2)

    # Threshold relative to each timestep's spatial mean (accounts for drift)
    mean_per_t = east_half.reshape(T, -1).mean(axis=1, keepdims=True)
    mean_per_t = mean_per_t[:, :, None]  # (T, 1, 1)
    above = east_half > (mean_per_t + contour_threshold_m)

    # For each timestep, find northernmost row index where any pixel is above
    # `above` shape: (T, H, W/2).  Find northernmost H index with any True.
    extent_lat = np.full(T, np.nan, dtype=np.float32)
    for t in range(T):
        # any True along lon axis
        row_has = above[t].any(axis=1)  # (H,)
        if row_has.any():
            northernmost = np.where(row_has)[0].max()
            extent_lat[t] = lat[northernmost]
    return extent_lat


def anomaly_magnitude(ssh_m, land_mask):
    """
    Domain-mean absolute SSH anomaly. A scalar 'activity' index per timestep.
    Anomaly is computed relative to the domain's time-mean at each pixel.

    Input:  ssh_m (T, H, W), land_mask (H, W)
    Output: mag   (T,) in meters
    """
    ocean = ~land_mask
    # Climatology: mean over time, per pixel
    clim = ssh_m.mean(axis=0, keepdims=True)  # (1, H, W)
    anom = ssh_m - clim  # (T, H, W)
    # Mean absolute anomaly over ocean pixels
    mag = np.abs(anom)[:, ocean].mean(axis=1)  # (T,)
    return mag.astype(np.float32)


def compute_all_regimes(ssh_norm, mean, std, lat, land_mask):
    """
    Given normalized SSH and metadata, compute all regime indicators.

    Returns dict with:
        eke:        (T, H, W)
        ow:         (T, H, W)
        lc_extent:  (T,)
        anom_mag:   (T,)
    """
    ssh_m = denormalize(ssh_norm, mean, std)
    print("  Computing EKE...")
    eke = eddy_kinetic_energy_proxy(ssh_m)
    print("  Computing Okubo-Weiss...")
    ow = okubo_weiss(ssh_m)
    print("  Computing Loop Current extent...")
    lc_extent = loop_current_northward_extent(ssh_m, lat)
    print("  Computing anomaly magnitude...")
    anom_mag = anomaly_magnitude(ssh_m, land_mask)

    # Zero out regime fields on land (they're meaningless there)
    eke[:, land_mask] = 0
    ow[:, land_mask] = 0

    return {
        "eke": eke,
        "ow": ow,
        "lc_extent": lc_extent,
        "anom_mag": anom_mag,
    }


if __name__ == "__main__":
    # Compute regimes on the test set (matches test_predictions.npz timesteps)
    import os
    print("Loading splits...")
    splits = np.load("data/processed/splits.npz")
    test = splits["test"]
    lat = splits["lat"]
    land_mask = splits["land_mask"]
    mean = float(splits["mean"])
    std = float(splits["std"])

    # The test predictions start at test index HISTORY+HORIZON-1 = 7
    # (because first prediction needs 7 days history)
    # So to align with targets we skip those first timesteps in regimes too.
    from dataset import HISTORY, HORIZON
    aligned_test = test[HISTORY + HORIZON - 1:]
    print(f"Computing regimes on {aligned_test.shape[0]} test timesteps...")

    regimes = compute_all_regimes(aligned_test, mean, std, lat, land_mask)

    out_path = "outputs/test_regimes.npz"
    os.makedirs("outputs", exist_ok=True)
    np.savez_compressed(out_path, **regimes)
    print(f"Saved {out_path}")
    for k, v in regimes.items():
        print(f"  {k}: shape {v.shape}, range [{np.nanmin(v):.4g}, {np.nanmax(v):.4g}]")
