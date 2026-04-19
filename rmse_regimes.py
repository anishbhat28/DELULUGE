"""
rmse_regimes.py
===============

Generic tabular regime extractor. Consumes a CSV of predictions and targets
and emits named regime fields that downstream tools (autoresearch.py) can
slice the error against.

Contract
--------
CSV must contain one target column and one prediction column. Column names
are auto-detected from a common-alias list, case-insensitive. Any additional
numeric columns are treated as features and exposed as regime fields
`feature::<colname>`.

Emitted regime fields (all length-N arrays aligned to the CSV rows):
    target            raw target value
    prediction        raw prediction value
    abs_error         |prediction - target|
    residual          prediction - target
    residual_sign     sign(residual)  (-1, 0, +1)
    feature::<col>    raw value of numeric feature column <col>
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


TARGET_ALIASES = ["target", "truth", "actual", "y", "y_true", "label", "ground_truth"]
PRED_ALIASES = ["prediction", "pred", "yhat", "y_hat", "y_pred", "predicted", "forecast"]


def _find_column(df: pd.DataFrame, aliases: list[str]) -> str | None:
    lowered = {c.lower(): c for c in df.columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def load_tabular(path: str | Path) -> dict:
    """Load a CSV and split into target / prediction / features."""
    path = Path(path)
    df = pd.read_csv(path)

    t_col = _find_column(df, TARGET_ALIASES)
    p_col = _find_column(df, PRED_ALIASES)
    if t_col is None or p_col is None:
        raise ValueError(
            f"CSV must contain a target column ({'/'.join(TARGET_ALIASES)}) "
            f"and a prediction column ({'/'.join(PRED_ALIASES)}). "
            f"Got columns: {list(df.columns)}"
        )

    targets = df[t_col].to_numpy(dtype=float)
    preds = df[p_col].to_numpy(dtype=float)
    features = df.drop(columns=[t_col, p_col]).select_dtypes(include=[np.number])

    residual = preds - targets
    return {
        "target": targets,
        "prediction": preds,
        "residual": residual,
        "abs_error": np.abs(residual),
        "features": features,
        "target_col": t_col,
        "pred_col": p_col,
        "n_rows": int(len(df)),
    }


def compute_regime_fields(bundle: dict) -> dict[str, np.ndarray]:
    """Map a loaded bundle to a flat dict of named regime arrays."""
    regimes = {
        "target": bundle["target"],
        "prediction": bundle["prediction"],
        "abs_error": bundle["abs_error"],
        "residual": bundle["residual"],
        "residual_sign": np.sign(bundle["residual"]).astype(np.int8),
    }
    for col in bundle["features"].columns:
        regimes[f"feature::{col}"] = bundle["features"][col].to_numpy(dtype=float)
    return regimes


def rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(errors ** 2)))


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
    bundle = load_tabular(path)
    regimes = compute_regime_fields(bundle)

    print(f"Loaded {bundle['n_rows']} rows from {path}")
    print(f"  target column     = {bundle['target_col']}")
    print(f"  prediction column = {bundle['pred_col']}")
    print(f"  feature columns   = {list(bundle['features'].columns)}")
    print(f"  RMSE              = {rmse(bundle['abs_error']):.6g}")
    print(f"  mean |error|      = {bundle['abs_error'].mean():.6g}")
    print("  regime fields:")
    for k in regimes:
        print(f"    {k}")

    os.makedirs("outputs", exist_ok=True)
    np.savez_compressed("outputs/test_regimes.npz", **regimes)
    print("Saved outputs/test_regimes.npz")


if __name__ == "__main__":
    main()
