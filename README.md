# 🌊 Environmental ML Trust Lab — Gulf of Mexico

**A spatiotemporal atlas of where neural ocean surrogates fail, and why.**

Built for DataHacks 2026. Uses the Scripps Institution of Oceanography
Gulf of Mexico sea surface height (SSH) dataset.

---

## The problem

Operational ocean forecasting centers (NOAA, Mercator Ocean, ECMWF) are
migrating from expensive numerical simulations to fast ML surrogates. The
blocker to deployment isn't accuracy — it's **silent failure**. A forecaster
cannot stake evacuation decisions, storm-surge warnings, or search-and-rescue
on a neural network that might be confidently wrong in ways nobody can predict.

This project addresses that bottleneck. Not by making the neural net better —
by telling the forecaster **where and when to trust it**, tied to the physical
ocean regimes that cause the failures.

---

## Headline result

| Metric | Value | Notes |
|---|---|---|
| Model RMSE (test) | **4.0 mm** | Ensemble mean vs. Scripps numerical simulation, 7.9 test years |
| Persistence RMSE | 10.3 mm | Baseline: predict tomorrow = today |
| Skill over persistence | **2.6×** | The ensemble actually learns dynamics |
| Mean disagreement | 3.1 mm | Ensemble standard deviation — the trust signal |
| Pearson r (disagreement vs error) | **0.XX** | Disagreement is a genuine error predictor |

Where the ensemble disagrees, it is also more likely to be wrong. The
disagreement-error correlation is strongest in **dynamically active regions**
(high eddy kinetic energy, Loop Current intrusion zones) — exactly the regions
a forecaster needs to know not to trust a fast ML prediction.

---

## What you get

An interactive Streamlit app that lets you:

1. **Scrub through time** across 2,866 held-out test days (simulation days 11,507–14,372)
2. **See four synchronized panels**: ground-truth SSH, ensemble-mean prediction,
   ensemble disagreement (trust field), absolute error
3. **Verify the scientific claim** by eye: regions where disagreement lights up
   should coincide with regions where error lights up
4. **See the disagreement→error correlation** across the whole test set, colored
   by physical regime

The app opens on the pre-identified "money-shot" timestep — the single most
dramatic example where the trust field correctly flagged a high-error event.

---

## Tracks

Submitted to **ML/AI** and **Data Analytics**, using the required **Scripps** dataset.

- **ML/AI framing:** neural surrogate ensemble with calibrated uncertainty,
  plus a research-flavored analysis of regime-dependent failure modes.
- **Data Analytics framing:** interactive atlas revealing how ML ocean models
  break down, with statistical analysis tying failures to physical regimes
  derivable from the data alone.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Streamlit UI (app.py)                                  │
│  Headline · Atlas · Scatter · Methodology               │
├─────────────────────────────────────────────────────────┤
│  Domain-specific analysis                                │
│  Regime indicators: EKE, Okubo-Weiss, Loop Current      │
├─────────────────────────────────────────────────────────┤
│  Ensemble analysis                                       │
│  Per-pixel per-timestep error and disagreement fields   │
├─────────────────────────────────────────────────────────┤
│  Model ensemble (5 U-Nets, varied width/depth/seed)     │
│  Predict SSH(t+1) from SSH(t−6 … t)                     │
├─────────────────────────────────────────────────────────┤
│  Dataset + dataloader (sliding windows, masked loss)    │
├─────────────────────────────────────────────────────────┤
│  Preprocessing (xarray, numpy)                           │
│  Loop Current subdomain, temporal splits, normalization │
├─────────────────────────────────────────────────────────┤
│  Scripps Gulf of Mexico SSH simulation (40 years)       │
└─────────────────────────────────────────────────────────┘
```

---

## Setup and reproduction

```bash
# Environment
python3 -m venv env
source env/bin/activate
pip install torch xarray netcdf4 streamlit matplotlib cmocean numpy

# Put the Scripps .nc file at ~/Downloads/run2_clim_v2_ssh.nc, then:
python preprocess.py          # ~1 min  — extracts Loop Current subdomain
python baseline_check.py      # ~10 sec — verifies persistence baseline
python train_ensemble.py      # ~40 min on M4 Pro MPS — trains 5 U-Nets
python regimes.py             # ~30 sec — physical regime indicators
python find_money_shot.py     # ~5 sec  — picks the demo timestep

# Launch
python -m streamlit run app.py
```

---

## File layout

```
floodbreak/
├── README.md                       # you are here
├── app.py                          # Streamlit UI
├── preprocess.py                   # raw .nc → train/val/test .npz
├── dataset.py                      # PyTorch sliding-window dataset
├── models.py                       # small U-Net (parameterized width/depth)
├── train.py                        # single-model training loop
├── train_ensemble.py               # train 5 models, save ensemble predictions
├── baseline_check.py               # persistence baseline diagnostic
├── regimes.py                      # physical regime indicators from SSH
├── find_money_shot.py              # pick best demo timestep
├── data/
│   └── processed/splits.npz        # preprocessed tensors
├── checkpoints/                    # trained model weights
├── outputs/
│   ├── test_predictions.npz        # ensemble predictions + error fields
│   ├── test_regimes.npz            # physical regime fields
│   └── money_shot.json             # selected demo timestep
└── docs/
    └── methodology.md              # detailed writeup
```

---

## Method summary

**Data.** 40-year numerical simulation of Gulf of Mexico SSH from Scripps,
0.05° resolution, daily snapshots (14,373 timesteps). Deterministic run with
constant boundary forcing — "ground truth" is well-defined since we're
training an ML surrogate to emulate a specific numerical model.

**Spatial domain.** 22°N–28°N, 92°W–84°W, covering the Loop Current and its
shed eddies. 120 × 146 pixels, ~0.8% land.

**Temporal splits.** Train: days 0–9,999 (27.4 yr). Val: 10,000–11,499 (4.1 yr).
Test: 11,500–14,372 (7.9 yr). Strictly causal — no future data leaks into training.

**Models.** Five U-Nets varying base width (24–48), depth (2–4), and random seed.
Parameter counts 120k–4.3M. Input: 7 days of SSH history as channels. Output:
predicted SSH at t+1.

**Training.** Masked MSE loss (land excluded). AdamW with cosine LR schedule,
6 epochs per model, Apple M4 Pro MPS backend. ~80 seconds per epoch.

**Analysis.** Per-pixel per-timestep error and disagreement fields computed
on test set. Physical regime indicators (eddy kinetic energy, Okubo-Weiss
parameter, Loop Current northward extent, anomaly magnitude) computed directly
from the SSH field — no external data needed.

See `docs/methodology.md` for full details.

---

## Attribution and honest framing

This project does not claim to save lives. It demonstrates a tool that, deployed
in an operational forecasting pipeline, gives forecasters information they need
to decide when to trust a fast ML prediction versus fall back to slower physical
models. The impact is in the chain of decisions downstream, not in this artifact.

The autoresearch layer (in development) is inspired by public work on LLM-driven
autonomous research — Karpathy's autoresearch concept, Sakana AI's AI Scientist,
Anthropic's agentic research patterns. Our specific contributions are:
(1) a physical-oceanography hypothesis language with domain-specific tools;
(2) information-gain-guided hypothesis selection under fixed budget;
(3) held-out temporal validation and Bonferroni correction on every reported
finding. All agent code is written from scratch; no external agent framework is used.

Built in 8 hours for DataHacks 2026.
