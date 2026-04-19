"""
dashboard.py -- Streamlit dashboard for the ML Trust Lab (generic tabular).

Loads the uploaded CSV (data.csv) and the autoresearch output
(outputs/findings.json) and renders metrics, error diagnostics, and the
validated failure-mode report.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from rmse_regimes import load_tabular, rmse


st.set_page_config(
    page_title="ML Trust Lab",
    page_icon="🌊",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"], [data-testid="stSidebar"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)


DATA_CANDIDATES = ["predictions.csv", "data.csv"]
FINDINGS_PATH = "outputs/findings.json"


@st.cache_data
def load_bundle(path: str):
    return load_tabular(path)


def find_data_path() -> str | None:
    for cand in DATA_CANDIDATES:
        if os.path.exists(cand):
            return cand
    return None


# ---------- Header ----------
st.title("🌊 ML Trust Lab")


data_path = find_data_path()
if data_path is None:
    st.warning(
        "No `predictions.csv` or `data.csv` in the project root. Upload on the **Upload** page first — "
        "the frontend will save your upload, run your train.py to produce predictions.csv, then kick off autoresearch."
    )
    st.stop()

try:
    bundle = load_bundle(data_path)
except Exception as e:
    st.error(f"Could not load {data_path}: {e}")
    st.stop()

n = bundle["n_rows"]
abs_error = bundle["abs_error"]
feature_names = list(bundle["features"].columns)


# ---------- Headline metrics ----------
model_rmse = rmse(abs_error)
mae = float(np.mean(abs_error))

c1, c2, c3, c4 = st.columns(4)
c1.metric("RMSE", f"{model_rmse:.4g}", help=f"Root mean squared error of {bundle['pred_col']} vs {bundle['target_col']}")
c2.metric("MAE", f"{mae:.4g}", help="Mean absolute error")
c3.metric("Rows", f"{n:,}")
c4.metric("Feature columns", f"{len(feature_names)}")


# ---------- Atlas (spatial SSH view, ported from main branch) ----------
ATLAS_PREDS = "outputs/test_predictions.npz"
ATLAS_REGIMES = "outputs/test_regimes.npz"
if os.path.exists(ATLAS_PREDS):
    try:
        _atlas_preds = np.load(ATLAS_PREDS)
        _required = {"targets", "ensemble_mean", "ensemble_std", "abs_error",
                     "land_mask", "lat", "lon", "std_norm"}
        if _required.issubset(set(_atlas_preds.files)):
            try:
                import cmocean
                _cmap_ssh = cmocean.cm.balance
                _cmap_amp = cmocean.cm.amp
            except ImportError:
                _cmap_ssh = "RdBu_r"
                _cmap_amp = "Reds"

            _atlas_regimes = np.load(ATLAS_REGIMES) if os.path.exists(ATLAS_REGIMES) else None

            _atlas = {
                "targets": _atlas_preds["targets"],
                "ensemble_mean": _atlas_preds["ensemble_mean"],
                "ensemble_std": _atlas_preds["ensemble_std"],
                "abs_error": _atlas_preds["abs_error"],
                "land_mask": _atlas_preds["land_mask"],
                "lat": _atlas_preds["lat"],
                "lon": _atlas_preds["lon"],
                "norm_std": float(_atlas_preds["std_norm"]),
            }
            _atlas_ocean = ~_atlas["land_mask"]
            _T, _H, _W = _atlas["targets"].shape

            def _default_t():
                try:
                    with open("outputs/money_shot.json") as _f:
                        return int(json.load(_f)["timestep"])
                except Exception:
                    return _T // 2

            st.header("The atlas — drag the slider to explore")
            _t = st.slider(
                "Test-set day index",
                min_value=0, max_value=_T - 1, value=_default_t(), step=1,
            )

            def _mask_land(arr, land_mask):
                out = arr.copy().astype(float)
                out[land_mask] = np.nan
                return out

            _truth = _atlas["targets"][_t] * _atlas["norm_std"]
            _pred = _atlas["ensemble_mean"][_t] * _atlas["norm_std"]
            _disag = _atlas["ensemble_std"][_t] * _atlas["norm_std"]
            _err = _atlas["abs_error"][_t] * _atlas["norm_std"]
            _vmax_ssh = float(max(np.abs(_truth).max(), np.abs(_pred).max()))
            _vmax_disag = float(_disag[_atlas_ocean].max())
            _vmax_err = float(_err[_atlas_ocean].max())

            _fig_atlas, _axes = plt.subplots(1, 4, figsize=(18, 4.5))
            _extent = [_atlas["lon"].min(), _atlas["lon"].max(),
                       _atlas["lat"].min(), _atlas["lat"].max()]

            _im0 = _axes[0].imshow(_mask_land(_truth, _atlas["land_mask"]), cmap=_cmap_ssh,
                                   origin="lower", extent=_extent,
                                   vmin=-_vmax_ssh, vmax=_vmax_ssh, aspect="auto")
            _axes[0].set_title("Truth SSH (m)")
            plt.colorbar(_im0, ax=_axes[0], fraction=0.046, pad=0.04)

            _im1 = _axes[1].imshow(_mask_land(_pred, _atlas["land_mask"]), cmap=_cmap_ssh,
                                   origin="lower", extent=_extent,
                                   vmin=-_vmax_ssh, vmax=_vmax_ssh, aspect="auto")
            _axes[1].set_title("Ensemble mean prediction (m)")
            plt.colorbar(_im1, ax=_axes[1], fraction=0.046, pad=0.04)

            _im2 = _axes[2].imshow(_mask_land(_disag, _atlas["land_mask"]), cmap=_cmap_amp,
                                   origin="lower", extent=_extent,
                                   vmin=0, vmax=_vmax_disag, aspect="auto")
            _axes[2].set_title("Disagreement (m) -- trust field")
            plt.colorbar(_im2, ax=_axes[2], fraction=0.046, pad=0.04)

            _im3 = _axes[3].imshow(_mask_land(_err, _atlas["land_mask"]), cmap=_cmap_amp,
                                   origin="lower", extent=_extent,
                                   vmin=0, vmax=_vmax_err, aspect="auto")
            _axes[3].set_title("Absolute error (m)")
            plt.colorbar(_im3, ax=_axes[3], fraction=0.046, pad=0.04)

            for _ax in _axes:
                _ax.set_xlabel("Longitude")
                _ax.set_ylabel("Latitude")

            _fig_atlas.suptitle(f"Test day index t = {_t}   |   simulation day ~ {11500 + 7 + _t}",
                                fontsize=11, y=1.02)
            _fig_atlas.tight_layout()
            st.pyplot(_fig_atlas)
            st.caption(
                "Reading the atlas: left two panels should look nearly identical (truth vs. prediction). "
                "Third panel shows where the ensemble is uncertain; fourth shows where it's actually wrong. "
                "The scientific claim: these two fields should correlate. Drag the slider to verify."
            )

            if _atlas_regimes is not None and {"lc_extent", "anom_mag", "eke"}.issubset(set(_atlas_regimes.files)):
                _r1, _r2, _r3 = st.columns(3)
                _r1.metric(
                    "Loop Current extent (this frame)",
                    f"{_atlas_regimes['lc_extent'][_t]:.2f}°N",
                    help="Northernmost latitude where SSH exceeds a high-anomaly contour in the eastern domain",
                )
                _r2.metric(
                    "Domain anomaly magnitude (this frame)",
                    f"{_atlas_regimes['anom_mag'][_t]*1000:.1f} mm",
                    help="Mean absolute SSH anomaly over the ocean pixels",
                )
                _r3.metric(
                    "Mean EKE in frame (relative)",
                    f"{float(_atlas_regimes['eke'][_t, _atlas_ocean].mean()):.2e}",
                    help="Eddy kinetic energy proxy from SSH gradients",
                )
    except Exception as _atlas_err:
        st.caption(f"Atlas skipped: {_atlas_err}")


# ---------- Predictions vs targets ----------
st.header("Predictions vs targets")

rng = np.random.default_rng(0)
sample_size = min(5000, n)
sample_idx = rng.choice(n, size=sample_size, replace=False)

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(bundle["target"][sample_idx], bundle["prediction"][sample_idx],
           s=6, alpha=0.35, c=abs_error[sample_idx], cmap="viridis")
lo = float(min(bundle["target"].min(), bundle["prediction"].min()))
hi = float(max(bundle["target"].max(), bundle["prediction"].max()))
ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, linewidth=1, label="y = x")
ax.set_xlabel(f"{bundle['target_col']} (target)")
ax.set_ylabel(f"{bundle['pred_col']} (prediction)")
ax.set_title(f"n = {sample_size:,} sampled points  •  color = |error|")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
st.pyplot(fig)


# ---------- Error distribution ----------
st.header("Error distribution")

fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 4.5))
ax2a.hist(abs_error, bins=50, color="#8b6fd4", alpha=0.85)
ax2a.set_xlabel("|error|")
ax2a.set_ylabel("count")
ax2a.set_title("Absolute error histogram")

ax2b.hist(bundle["residual"], bins=50, color="#a894e0", alpha=0.85)
ax2b.axvline(0, color="k", linestyle="--", alpha=0.5)
ax2b.set_xlabel("residual (prediction − target)")
ax2b.set_ylabel("count")
ax2b.set_title("Residual histogram (bias check)")
fig2.tight_layout()
st.pyplot(fig2)

bias = float(np.mean(bundle["residual"]))
st.caption(f"Mean residual = {bias:+.4g}. Negative = model under-predicts on average; positive = over-predicts.")


# ---------- Validated failure modes ----------
st.header("Validated failure modes (autoresearch loop)")

if not os.path.exists(FINDINGS_PATH):
    st.info(
        "No findings.json yet. Run `python autoresearch.py --data data.csv` to populate this section. "
        "The loop will propose regime hypotheses, test them on a discovery split, and validate the "
        "survivors on a held-out split with Bonferroni correction."
    )
else:
    with open(FINDINGS_PATH) as _f:
        findings_doc = json.load(_f)

    findings = findings_doc.get("findings", [])
    n_total = len(findings)
    n_validated = sum(1 for f in findings if f.get("validated"))
    n_rejected = n_total - n_validated
    alpha = findings_doc.get("config", {}).get("bonferroni_alpha", 0.05)

    st.markdown(
        f"""
        The autoresearch loop proposed **{n_total} regime hypotheses**, tested them on a discovery
        split, and validated the survivors on a held-out split with Bonferroni correction at
        α = {alpha:.4f}. **{n_validated} passed** validation; **{n_rejected} were correctly rejected**
        when their effect failed to generalize.

        Every finding below carries a pair of 8-character receipt IDs pointing to the tool calls
        that computed each number. The full execution trace is in `outputs/findings.json`.
        """
    )

    def describe(f):
        r = f.get("regime_field", f.get("regime_type", "?"))
        c = f.get("comparator", "")
        v = f.get("value", "")
        if c == "percentile_gt":
            return f"`{r}` above {v:.1f}-th percentile"
        if c == "percentile_lt":
            return f"`{r}` below {v:.1f}-th percentile"
        if c in ("gt", "lt", "eq"):
            return f"`{r}` {c} {v}"
        return f"`{r}` {c} {v}"

    for f in findings:
        if not f.get("validated"):
            continue
        val, disc = f["validation"], f["discovery"]
        with st.container(border=True):
            st.markdown(f"**✓ VALIDATED — {describe(f)}**")
            cc1, cc2, cc3 = st.columns(3)
            cc1.metric("Error ratio (validation)", f"{val['error_ratio']:.2f}×",
                       help="Mean error inside regime / mean error outside, on held-out data")
            cc2.metric("Mean err inside (val)", f"{f['mean_err_inside_val']:.4g}")
            cc3.metric("Mean err outside (val)", f"{f['mean_err_outside_val']:.4g}")
            st.caption(
                f"p-value (validation): {val['p_value']:.2e}  •  "
                f"discovery err-ratio: {disc['error_ratio']:.2f}× (p={disc['p_value']:.2e})  •  "
                f"Receipts: discovery `{disc['call_id']}`, validation `{val['call_id']}`"
            )

    for f in findings:
        if f.get("validated"):
            continue
        val, disc = f["validation"], f["discovery"]
        with st.container(border=True):
            st.markdown(f"**✗ REJECTED — {describe(f)}**")
            cc1, cc2 = st.columns(2)
            cc1.metric("Discovery err ratio", f"{disc['error_ratio']:.2f}×",
                       help="On the split the agent could query")
            cc2.metric(
                "Validation err ratio",
                f"{val['error_ratio']:.2f}×",
                delta=f"effect {'reversed' if val['error_ratio'] < 1 else 'shrank'}",
                delta_color="inverse",
                help="On the held-out split the agent never saw",
            )
            st.caption(
                f"Correctly rejected by Bonferroni-corrected holdout. "
                f"Receipts: discovery `{disc['call_id']}`, validation `{val['call_id']}`."
            )


# ---------- Method notes ----------
st.header("Method notes")
st.markdown(
    """
    **Discovery / validation split.** Rows are randomly permuted with a fixed seed and split 70/30.
    The agent can only query the discovery split; every candidate is re-tested on the held-out
    validation split with Bonferroni correction to control the family-wise error rate.

    **Regime language.** Generic tabular: target/prediction percentiles, absolute-error percentiles,
    residual sign (over- vs under-prediction), and feature-column percentiles/thresholds. The agent
    enumerates these and tests Welch's t-test on inside-vs-outside absolute error.

    **Receipts.** Every finding links back to its tool-call id, so no reported number is
    unchallengeable — the full execution trace is persisted to `outputs/findings.json`.
    """
)
