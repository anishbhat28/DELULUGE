# Methodology

This document describes the technical and scientific details of the
Environmental ML Trust Lab.

---

## 1. Scientific motivation

Geophysical fluid dynamics simulations (ocean, atmosphere) are computationally
expensive. A single 40-year run of a regional ocean model like the one used
here takes weeks on a supercomputer. Operational forecasting needs predictions
in minutes, not weeks, which is why the field has been investing heavily in
neural surrogates (GraphCast, FourCastNet, Aurora, neural operator methods).

The practical blocker is calibration. Operational meteorology has decades of
experience calibrating physical models (ensemble Kalman filtering, probabilistic
post-processing), but neural surrogates are harder to trust: they fail silently
and their failures are often localized in specific regions or regimes that
aggregate metrics hide.

Our contribution is a methodology for **mapping** where and when a trained
surrogate fails, tied to physically interpretable regimes.

---

## 2. Data

The Scripps Gulf of Mexico sea surface height (SSH) dataset is a 40-year
numerical simulation with constant boundary forcing. Because the boundary
forcing is constant, the dataset is effectively a long realization of the
Gulf's intrinsic variability — driven by the Loop Current and the eddies
it sheds, rather than by interannual forcing changes.

- Variable: SSH, meters
- Grid: 0.05° × 0.05° (~5.5 km × 5 km at 25°N)
- Timesteps: 14,373 daily snapshots
- Full domain: 8.55°N–30.85°N, 97.95°W–72.45°W
- File: `run2_clim_v2_ssh.nc`, 2.05 GB

### 2.1 Subdomain selection

We extract a 120 × 146 subdomain covering 22°N–28°N, 92°W–84°W. This window
contains the canonical Loop Current path and its shed eddies. Outside this
region the Gulf is substantially quieter and less relevant for the
physics-informed failure analysis.

After extraction, 0.8% of pixels are land (Yucatan peninsula and Florida shelf
edges). We represent land as NaN in the raw data and zero in the normalized
fields, and use an explicit boolean land-mask in the loss function.

### 2.2 Temporal splits

| Split | Days | Years | Use |
|---|---|---|---|
| Train | 0–9,999 | 27.4 | Model training, normalization statistics |
| Val | 10,000–11,499 | 4.1 | Checkpoint selection, baseline comparison |
| Test | 11,500–14,372 | 7.9 | Ensemble predictions, atlas, analysis |

Splits are strictly chronological to prevent temporal leakage.

### 2.3 Normalization

We compute the mean and standard deviation of SSH across the training split,
ocean pixels only, and apply the same statistics to val and test. On the
Loop Current subdomain: **mean = -0.09 m, std = 0.23 m**. The 23-cm standard
deviation reflects the full dynamic range of Loop Current eddies, which is
consistent with altimetric observations of the real Gulf.

---

## 3. Model

We train a small U-Net to predict SSH at time t+1 from SSH at times t−6 … t
(seven days of history as input channels).

### 3.1 Architecture

A standard U-Net with configurable width and depth:

- Encoder: `depth` levels of (Conv3×3 → GroupNorm → ReLU) × 2, then MaxPool2
- Bottleneck: same block at depth+1
- Decoder: ConvTranspose2 upsampling with skip concatenation, then (Conv×2) block
- Output head: 1×1 conv → 1 channel

### 3.2 Ensemble configuration

Five members with varied width, depth, and seed:

| ID | base_width | depth | seed | params |
|---|---|---|---|---|
| m0 | 32 | 3 | 0 | 1.93M |
| m1 | 32 | 3 | 1 | 1.93M |
| m2 | 48 | 3 | 2 | 4.34M |
| m3 | 24 | 4 | 3 | 3.76M |
| m4 | 32 | 2 | 4 | 0.50M |

The width/depth/seed diversity ensures that ensemble disagreement reflects
**model uncertainty** (different function approximations of the same underlying
dynamics) rather than just random initialization noise.

### 3.3 Loss

Masked MSE: land pixels contribute zero to the loss. The mask is applied at
batch level rather than precomputed, which is marginally more expensive but
handles edge cases cleanly.

### 3.4 Optimization

- Optimizer: AdamW, weight decay 1e-4
- Learning rate: 2e-4 – 4e-4 depending on member (cosine schedule over epochs)
- Batch size: 16
- Epochs: 6
- Gradient clipping: max-norm 1.0
- Hardware: Apple M4 Pro, MPS backend
- Per-epoch time: ~80 seconds

---

## 4. Baselines

### 4.1 Persistence

The simplest baseline for a slowly-varying field: predict SSH(t+1) = SSH(t).
For daily SSH in the Gulf, the day-to-day change is small (~1 cm RMS), so
persistence is a strong baseline.

On our validation set, persistence RMSE = 10.3 mm.

### 4.2 Zero prediction

Predicting the climatological mean (after normalization, zero) gives MSE ≈ 1.0
in normalized units, as expected from z-score normalization.

### 4.3 Our ensemble

Ensemble-mean RMSE on the held-out test set: 4.0 mm.

The 2.6× improvement over persistence is the evidence that the model learns
dynamics, not just copies the input.

---

## 5. Analysis

### 5.1 Disagreement and error fields

For each test timestep, we compute:

- **Ensemble mean**: arithmetic mean of the 5 members' predictions
- **Disagreement**: per-pixel standard deviation of the 5 members
- **Error**: absolute difference between ensemble mean and ground truth

All three are per-pixel per-timestep fields with shape (2866, 120, 146).

### 5.2 Disagreement-as-uncertainty hypothesis

A well-calibrated ensemble should have its disagreement predict its error:
where the ensemble knows it doesn't know, it should be more likely to be wrong.

We test this by computing the Pearson correlation between disagreement and
absolute error across all ocean pixels and all test timesteps. A positive
correlation validates the ensemble's ability to self-diagnose.

### 5.3 Physical regime indicators

All regime indicators are derived from the SSH field itself. We use:

**Eddy kinetic energy (EKE) proxy.** Geostrophic velocity in a rotating reference
frame is proportional to SSH gradients:

  u ∝ −∂SSH/∂y,  v ∝ ∂SSH/∂x

EKE is then proportional to |∇SSH|². We use this proportional quantity as a
relative indicator (higher values = more energetic region).

**Okubo-Weiss parameter.** A standard oceanographic diagnostic distinguishing
vortex-dominated regions (OW < 0, inside eddy cores) from strain-dominated
regions (OW > 0, between eddies, at fronts):

  OW = S_n² + S_s² − W²

where S_n and S_s are the normal and shear strain rates and W is the relative
vorticity, all computed from the geostrophic velocity field.

**Loop Current northward extent.** Per-timestep scalar indicating how far north
the Loop Current pushes. Computed as the northernmost latitude where SSH in
the eastern half of the domain exceeds the spatial mean by a threshold.

**Anomaly magnitude.** Domain-mean absolute SSH anomaly from the per-pixel
climatology. A scalar "activity" index per timestep.

### 5.4 Regime-dependent error analysis

We bin test pixels by their local regime indicators and compute the mean
disagreement and mean error in each bin. The central finding is that
high-EKE and vortex-core (OW < 0) pixels have disproportionately high
error and disagreement — the model struggles most where the ocean is
dynamically active.

This is the physically interpretable failure mode the lab is designed to surface.

---

## 6. Money-shot selection

For the demo, we want a single timestep that dramatically illustrates the
thesis: disagreement flagged a region correctly, and the error was indeed
there. We compute a composite score per timestep:

  score(t) = z(mean_error_t) + z(mean_disag_t) + 0.5·z(log EKE_t)

where z(·) is z-score normalization across the test set. The highest-scoring
timestep is chosen as the default slider position in the demo.

This is a curation choice, not a cherry-pick — we show a real, high-signal
example from the test set and let users scrub to verify the pattern holds
elsewhere.

---

## 7. Autoresearch layer (in development)

The analysis above is done by the authors by hand. The autoresearch layer
automates hypothesis generation by giving an LLM agent:

1. **A narrow, typed tool interface**: `compute_eddy_field`, `compute_loop_current_position`,
   `evaluate_in_mask`, `correlate`, `propose_regime`
2. **A physical-oceanography hypothesis language**: hypotheses are expressed
   as region masks defined in terms of the regime indicators, not generic
   Python code
3. **A held-out validation protocol**: each proposed regime is tested against
   data the agent did not query during hypothesis generation
4. **A multiple-testing correction**: Bonferroni-corrected significance levels
   account for the number of hypotheses considered

Every finding surfaces with the tool-call IDs that produced it, so a reader
can trace each claim to an executed computation. This is the specific sense
in which the system is "grounded" — no number in a claim comes from LLM
generation; every number comes from code execution.

---

## 8. Limitations

- The spatial domain is one region of one basin; generalization to other
  dynamical regimes has not been tested.
- The numerical simulation has constant boundary forcing, so the dataset
  lacks interannual/decadal variability. Real ocean data would be harder.
- The test set is 7.9 years of a single simulation run, so statistics on
  rare events (large eddy-shedding events) are noisy.
- Persistence is a strong baseline for 1-day SSH forecasts; the skill gain
  reported here is meaningful but shrinks with longer lead times (not tested).

---

## 9. References and inspirations

- Chelton, D. B., et al. (2011). *Global observations of nonlinear mesoscale eddies.* Progress in Oceanography.
- Lakshminarayanan, B., et al. (2017). *Simple and scalable predictive uncertainty estimation using deep ensembles.* NeurIPS.
- Ronneberger, O., et al. (2015). *U-Net: Convolutional networks for biomedical image segmentation.*
- Kochkov, D., et al. (2021). *Machine-learning accelerated computational fluid dynamics.* PNAS.
- Karpathy, A. Autoresearch concept, public lectures and writing.
- Lu, C., et al. (2024). *The AI Scientist: Towards fully automated open-ended scientific discovery.* Sakana AI.
