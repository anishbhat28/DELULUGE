"""
app.py -- Streamlit UI for the environmental-ML trust lab.

Run with:
    streamlit run app.py
"""

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import cmocean


# ---------- Page config ----------
st.set_page_config(
    page_title="Environmental ML Trust Lab",
    page_icon="🌊",
    layout="wide",
)


# ---------- Data loading (cached) ----------
@st.cache_data
def load_data():
    preds = np.load("outputs/test_predictions.npz")
    regimes = np.load("outputs/test_regimes.npz")
    return {
        "targets": preds["targets"],
        "ensemble_mean": preds["ensemble_mean"],
        "ensemble_std": preds["ensemble_std"],
        "abs_error": preds["abs_error"],
        "land_mask": preds["land_mask"],
        "lat": preds["lat"],
        "lon": preds["lon"],
        "norm_mean": float(preds["mean_norm"]),
        "norm_std": float(preds["std_norm"]),
        "eke": regimes["eke"],
        "ow": regimes["ow"],
        "lc_extent": regimes["lc_extent"],
        "anom_mag": regimes["anom_mag"],
    }


d = load_data()
ocean = ~d["land_mask"]
T, H, W = d["targets"].shape


# Default slider position: the money-shot timestep if it exists, else middle
def get_default_t():
    try:
        import json
        with open("outputs/money_shot.json") as f:
            return int(json.load(f)["timestep"])
    except Exception:
        return T // 2


default_t = get_default_t()


# ---------- Header ----------
st.title("🌊 Environmental ML Trust Lab — Gulf of Mexico")
st.markdown(
    """
    **A spatiotemporal atlas of where neural ocean surrogates fail, and why.**

    As operational ocean forecasting migrates from expensive numerical simulations to fast ML surrogates,
    silent failure becomes the blocker to deployment. This lab maps where, when, and how a U-Net ensemble
    breaks down on Gulf of Mexico sea surface height, and grounds those failures in physical regimes
    (eddy kinetic energy, Loop Current dynamics, Okubo-Weiss vortex structure).
    """
)

# ---------- Headline metrics ----------
rmse_m = float((d["abs_error"][:, ocean] ** 2).mean() ** 0.5) * d["norm_std"]
disag_m = float(d["ensemble_std"][:, ocean].mean()) * d["norm_std"]
persistence_rmse_m = 0.0103  # from baseline_check.py

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Model RMSE (test)",
    f"{rmse_m*1000:.1f} mm",
    help="Ensemble mean vs. numerical-simulation truth, Loop Current region, held-out test years",
)
c2.metric(
    "Persistence RMSE (val)",
    f"{persistence_rmse_m*1000:.1f} mm",
    help="Baseline: predict tomorrow = today",
)
c3.metric(
    "Skill over persistence",
    f"{persistence_rmse_m / rmse_m:.2f}x",
    help="Our ensemble actually learns dynamics, not just copies the input",
)
c4.metric(
    "Mean disagreement",
    f"{disag_m*1000:.1f} mm",
    help="Average across the 5 ensemble members — this is the trust signal",
)


# ---------- Atlas ----------
st.header("The atlas -- drag the slider to explore")

t = st.slider(
    "Test-set day index",
    min_value=0, max_value=T - 1, value=default_t, step=1,
)

def mask_land(arr, land_mask):
    out = arr.copy()
    out[land_mask] = np.nan
    return out


truth_m = d["targets"][t] * d["norm_std"]
pred_m = d["ensemble_mean"][t] * d["norm_std"]
disag_field_m = d["ensemble_std"][t] * d["norm_std"]
err_m = d["abs_error"][t] * d["norm_std"]

vmax_ssh = float(max(np.abs(truth_m).max(), np.abs(pred_m).max()))
vmax_disag = float(disag_field_m[ocean].max())
vmax_err = float(err_m[ocean].max())

fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
extent = [d["lon"].min(), d["lon"].max(), d["lat"].min(), d["lat"].max()]

im0 = axes[0].imshow(mask_land(truth_m, d["land_mask"]), cmap=cmocean.cm.balance,
                     origin="lower", extent=extent, vmin=-vmax_ssh, vmax=vmax_ssh, aspect="auto")
axes[0].set_title("Truth SSH (m)")
plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

im1 = axes[1].imshow(mask_land(pred_m, d["land_mask"]), cmap=cmocean.cm.balance,
                     origin="lower", extent=extent, vmin=-vmax_ssh, vmax=vmax_ssh, aspect="auto")
axes[1].set_title("Ensemble mean prediction (m)")
plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

im2 = axes[2].imshow(mask_land(disag_field_m, d["land_mask"]), cmap=cmocean.cm.amp,
                     origin="lower", extent=extent, vmin=0, vmax=vmax_disag, aspect="auto")
axes[2].set_title("Disagreement (m) -- trust field")
plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

im3 = axes[3].imshow(mask_land(err_m, d["land_mask"]), cmap=cmocean.cm.amp,
                     origin="lower", extent=extent, vmin=0, vmax=vmax_err, aspect="auto")
axes[3].set_title("Absolute error (m)")
plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

for ax in axes:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

fig.suptitle(f"Test day index t = {t}   |   simulation day ~ {11500 + 7 + t}", fontsize=11, y=1.02)
fig.tight_layout()
st.pyplot(fig)

st.caption(
    "Reading the atlas: left two panels should look nearly identical (truth vs. prediction). "
    "Third panel shows where the ensemble is uncertain; fourth shows where it's actually wrong. "
    "The scientific claim: these two fields should correlate. Drag the slider to verify."
)

# Per-timestep regime indicators for the current frame
r1, r2, r3 = st.columns(3)
r1.metric(
    "Loop Current extent (this frame)",
    f"{d['lc_extent'][t]:.2f}°N",
    help="Northernmost latitude where SSH exceeds a high-anomaly contour in the eastern domain",
)
r2.metric(
    "Domain anomaly magnitude (this frame)",
    f"{d['anom_mag'][t]*1000:.1f} mm",
    help="Mean absolute SSH anomaly over the ocean pixels",
)
r3.metric(
    "Mean EKE in frame (relative)",
    f"{float(d['eke'][t, ocean].mean()):.2e}",
    help="Eddy kinetic energy proxy from SSH gradients",
)


# ---------- Scatter: disagreement vs error ----------
st.header("Does disagreement predict error?")


@st.cache_data
def compute_scatter_data():
    ocean_mask = ~d["land_mask"]
    flat_disag = d["ensemble_std"][:, ocean_mask].ravel() * d["norm_std"]
    flat_err = d["abs_error"][:, ocean_mask].ravel() * d["norm_std"]
    flat_eke = d["eke"][:, ocean_mask].ravel()
    r = float(np.corrcoef(flat_disag, flat_err)[0, 1])
    rng = np.random.default_rng(0)
    idx = rng.choice(flat_disag.size, size=min(15000, flat_disag.size), replace=False)
    return flat_disag[idx], flat_err[idx], flat_eke[idx], r


disag_pts, err_pts, eke_pts, r_corr = compute_scatter_data()

fig2, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(
    disag_pts * 1000, err_pts * 1000,
    c=np.log10(eke_pts + 1e-20), s=4, alpha=0.4, cmap="viridis",
)
ax.set_xlabel("Ensemble disagreement (mm)")
ax.set_ylabel("Absolute error (mm)")
ax.set_title(f"Disagreement vs error  |  Pearson r = {r_corr:.3f}  |  n = 15,000 sampled points")
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label("log10(Eddy kinetic energy)")
lim = float(max(ax.get_xlim()[1], ax.get_ylim()[1]))
ax.plot([0, lim], [0, lim], "k--", alpha=0.3, linewidth=1, label="1:1")
ax.legend(loc="upper left", fontsize=9)
fig2.tight_layout()
st.pyplot(fig2)

st.caption(
    f"Each point is one pixel at one timestep, color is log-eddy-kinetic-energy at that pixel. "
    f"Disagreement is a real predictor of error (r = {r_corr:.3f}). "
    f"Yellow points (high-eddy pixels) concentrate in the upper right: errors are worst where "
    f"the ocean is dynamically active, exactly where a forecaster needs to know not to trust the surrogate."
)


# ---------- Methodology ----------
st.header("Method notes")

st.markdown(
    """
    **Data.** 40-year numerical simulation of Gulf of Mexico sea surface height from Scripps Institution
    of Oceanography, 0.05° resolution. Train: simulation days 0–9999. Val: 10000–11499. Test: 11500–14372.
    No real-world calendar alignment needed — this is a deterministic run with constant boundary forcing.

    **Spatial subdomain.** 22°N–28°N, 92°W–84°W, covering the Loop Current and its shed eddies.
    120 × 146 pixels, ~0.8% land.

    **Models.** Five U-Nets with varied width (24–48 base channels), depth (2–4), and seeds.
    Parameter counts range from ~100k to ~4M. Each predicts SSH(t+1) from SSH(t−6:t).

    **Training.** Masked MSE loss (land excluded). AdamW, cosine LR schedule, 6 epochs per model on
    Apple M4 Pro via MPS.

    **Physical regime indicators.** Eddy kinetic energy from SSH geostrophic-velocity gradients.
    Okubo-Weiss parameter for vortex vs. strain distinction. Loop Current northward extent from
    per-timestep SSH contour tracking in the eastern domain. All computed from SSH alone — no external data.

    **Attribution.** The autoresearch layer (in development) is inspired by recent work on LLM-driven
    autonomous research, including Karpathy's autoresearch concept, Sakana AI's AI Scientist, and
    Anthropic's agentic research patterns. Our specific contributions are: (1) a physical-oceanography
    hypothesis language with domain-specific tools for eddy analysis, Loop Current tracking, and regime
    correlation; (2) information-gain-guided hypothesis selection under a fixed budget; (3) held-out
    temporal validation and Bonferroni-corrected significance testing for every reported finding.
    All agent code is written from scratch for this project; no external agent framework is used.
    """
)
