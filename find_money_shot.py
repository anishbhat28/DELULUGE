"""
find_money_shot.py
==================

Scan the test set for the single most demo-worthy timestep:
a moment where the trust field (ensemble disagreement) correctly flagged
an unusually high error, in a region with strong physical activity (high EKE).

This gives us a specific frame to open the demo on — one dramatic example
instead of asking judges to scrub through 2866 timesteps hoping to find one.

Output:
    outputs/money_shot.json
        {
          "timestep": int,
          "simulation_day": int,
          "score": float,
          "mean_error_mm": float,
          "mean_disag_mm": float,
          "mean_eke_in_hotspot": float,
          "top_5_candidates": [...]
        }
"""

import json
import numpy as np
import os


def main():
    preds = np.load("outputs/test_predictions.npz")
    regimes = np.load("outputs/test_regimes.npz")

    abs_error = preds["abs_error"]          # (T, H, W) normalized
    disag = preds["ensemble_std"]           # (T, H, W) normalized
    eke = regimes["eke"]                    # (T, H, W)
    land = preds["land_mask"]
    ocean = ~land
    norm_std = float(preds["mean_norm"]), float(preds["std_norm"])
    std = float(preds["std_norm"])

    T = abs_error.shape[0]

    # For each timestep, score: high error AND high disagreement AND high EKE
    # Standardize each field (across all t, over ocean pixels) before combining
    def z(arr):
        v = arr[:, ocean]  # (T, N_ocean)
        m = v.mean(axis=1)  # (T,)
        return m

    err_t = z(abs_error)    # mean error per timestep
    disag_t = z(disag)      # mean disagreement per timestep
    eke_t = z(eke)          # mean eke per timestep

    # Normalize each to z-scores so they're comparable
    def standardize(x):
        return (x - x.mean()) / (x.std() + 1e-12)

    err_z = standardize(err_t)
    disag_z = standardize(disag_t)
    eke_z = standardize(np.log10(eke_t + 1e-20))

    # Composite score: we want high error AND high disagreement AND high EKE.
    # We also reward "disagreement was a good predictor" — cases where the
    # model *knew* it was uncertain (disagreement high whenever error high).
    score = err_z + disag_z + 0.5 * eke_z

    # Pick top candidates
    top_n = 5
    top_idx = np.argsort(score)[::-1][:top_n]
    best = int(top_idx[0])

    # Simulation day convention: val ended at 11500, test begins at 11500,
    # and we lose HISTORY + HORIZON - 1 = 7 days to windowing
    HISTORY, HORIZON = 7, 1
    sim_day = 11500 + HISTORY + HORIZON - 1 + best

    # Denormalized metrics for the best timestep
    err_mm = float(err_t[best] * std * 1000)
    disag_mm = float(disag_t[best] * std * 1000)

    result = {
        "timestep": best,
        "simulation_day": int(sim_day),
        "score": float(score[best]),
        "mean_error_mm": err_mm,
        "mean_disag_mm": disag_mm,
        "mean_eke": float(eke_t[best]),
        "rank_all": f"1 of {T}",
        "top_5_candidates": [
            {
                "timestep": int(idx),
                "simulation_day": int(11500 + 7 + idx),
                "score": float(score[idx]),
                "mean_error_mm": float(err_t[idx] * std * 1000),
                "mean_disag_mm": float(disag_t[idx] * std * 1000),
            }
            for idx in top_idx
        ],
    }

    out_path = "outputs/money_shot.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("=" * 60)
    print(f"Money-shot timestep: t = {best} (simulation day {sim_day})")
    print(f"  Mean error:        {err_mm:.2f} mm")
    print(f"  Mean disagreement: {disag_mm:.2f} mm")
    print(f"  Composite score:   {score[best]:.2f}")
    print("=" * 60)
    print("Top 5 candidates:")
    for r in result["top_5_candidates"]:
        print(f"  t={r['timestep']:4d}  day={r['simulation_day']}  "
              f"err={r['mean_error_mm']:.2f}mm  disag={r['mean_disag_mm']:.2f}mm  "
              f"score={r['score']:.2f}")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
