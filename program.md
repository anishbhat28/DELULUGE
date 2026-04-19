# Mission

Improve `train.py` through small, reversible experiments to better answer the user question: does the model accurately predict SSH levels?

Treat this as a regression problem on `target` (very likely SSH). Prioritize improving out-of-sample prediction quality and making evaluation trustworthy.

# Primary objective

Optimize test-set regression quality for `target`, using the existing train/test split already present in `train.py`.

Because the current code clearly fits:
- an `XGBRegressor` on `train[XGB_FEATURES] -> train["target"]`
- an LSTM on sequence windows from `df[LSTM_FEATURES] -> df["target"]`

the most likely useful work is:
1. verify evaluation is measuring SSH prediction accurately
2. tighten validation/test methodology
3. make small feature/preprocessing/model adjustments
4. keep only changes that improve held-out performance fairly

# Files to edit

Edit:
- `train.py`

Do not edit unless absolutely required by a narrowly justified experiment:
- dataset files
- unrelated project files
- output parsing / hidden harness behavior

Avoid broad refactors. Prefer local, surgical changes.

# What is visible in the current codebase

## Current models
- XGBoost regressor:
  - `n_estimators=300`
  - `max_depth=4`
  - `learning_rate=0.05`
  - `subsample=0.8`
  - `random_state=42`
  - uses `eval_set=[(X_test_xgb, y_test)]`
- LSTM:
  - stacked LSTM(64, return_sequences=True) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(1)
  - loss = `"mse"`
  - optimizer = `"adam"`
  - `epochs=200`, `batch_size=32`
  - `validation_split=0.1`
  - early stopping is present

## Available raw features
Observed columns:
- `time`
- `latitude`
- `longitude`
- `adt`
- `ice_extent`
- `ice_missing`

Target:
- `target` (likely SSH)

## Likely constraints / caveats
- There is already custom inspection logic around arrays / target keys near lines 115–139, suggesting results may be auto-read by a harness. Preserve expected variable names and basic flow.
- The code likely runs on CPU unless otherwise configured. Keep experiments cheap.
- LSTM uses sequence windows built from the full dataframe; sequence splitting may be vulnerable to leakage or poor temporal validation if not handled carefully.
- `validation_split=0.1` on sequence data may shuffle or create non-time-aware validation unless explicitly disabled. Be conservative here.
- XGBoost currently evaluates directly on the test set; if you use early stopping or model selection, do not tune repeatedly against the test set without noting it.

# Success criteria

A change is better only if it improves held-out predictive accuracy for SSH (`target`) under a fair comparison.

Use at least these metrics:
- RMSE
- MAE
- R²

If easy, also log:
- correlation between prediction and target
- baseline comparison vs simple persistence / mean baseline

Prioritize RMSE as the main optimization metric unless the existing code prints a different official metric. Use MAE and R² as supporting checks.

# Required evaluation policy

## Fair comparison
For any experiment:
- keep the same train/test split as baseline
- keep random seeds fixed where applicable
- compare against an unchanged baseline run
- change one main thing at a time

## Minimum reporting
For every run, record:
- experiment name
- model(s) affected: XGB / LSTM / both
- exact code change
- train/test sizes if visible
- feature set used
- RMSE / MAE / R² on test
- runtime rough note
- whether kept or reverted

## Keep / revert rule
Keep a change only if:
- test RMSE improves meaningfully, and
- MAE/R² do not materially worsen, and
- the change does not introduce obvious leakage or instability

Revert if:
- test RMSE worsens
- gains are tiny and noisy with added complexity
- evaluation becomes less trustworthy
- runtime increases heavily without clear metric benefit

Conservative threshold:
- keep if RMSE improves by ~1% or more with no major downside
- if improvement is <1%, prefer simpler code unless repeated runs clearly support it

# First task: verify whether the model accurately predicts SSH levels

Before tuning, make evaluation explicit and easy to inspect.

Add or confirm code that prints, for both XGB and LSTM if available:
- RMSE
- MAE
- R²
- prediction mean/std vs target mean/std
- first few predictions vs truths
- optional residual summary

If there is already a final blended prediction, also print the same metrics for the blend.

This is the fastest way to answer the user’s request.

# Most likely optimization targets in this code

## 1. Evaluation reliability
Highest priority. The current script trains on one split and may not clearly report robust regression metrics. Improve clarity first.

## 2. Time-aware validation for LSTM
Because features include `time` and the model is sequence-based, validation must respect temporal order. Random or implicit validation on sequence data is risky.

## 3. Feature selection / preprocessing
Likely useful features:
- `adt`
- `ice_extent`
- `ice_missing`
- `time`
- possibly `latitude`, `longitude`

Potential issue:
- `latitude` and `longitude` may be constant or near-constant within a local track/series; check variance before assuming usefulness.
- `time` may need scaling or cyclic treatment, but start simple.

## 4. XGB hyperparameters
The current XGB setup is modest and probably under-tuned. Small adjustments here are cheap and high-signal.

## 5. LSTM input scaling
Neural nets generally benefit from feature scaling, especially with mixed-scale columns like `time`, spatial coordinates, and environmental variables.

# Safe modification priorities

## Priority A: Add strong regression reporting
Implement test metrics cleanly for each model output:
- `rmse = sqrt(mean_squared_error(y_true, y_pred))`
- `mae = mean_absolute_error(y_true, y_pred)`
- `r2 = r2_score(y_true, y_pred)`

Also add:
- naive baseline:
  - mean predictor baseline
  - if sequence context exists, persistence baseline using previous target where feasible
Compare model RMSE against these.

If the model cannot beat naive baselines, focus on validation/preprocessing before architecture changes.

## Priority B: Inspect target distribution and prediction calibration
Log:
- `y_test.mean(), y_test.std()`
- `pred.mean(), pred.std()`
A model that predicts the right average but low variance is not accurately tracking SSH levels.

## Priority C: Ensure no temporal leakage in LSTM
Review sequence creation and train/validation/test boundaries.

Preferred conservative policy:
- create sequences separately for train and test partitions
- for LSTM validation, use the last portion of training sequences as validation
- avoid random validation split on temporal sequences
- set `shuffle=False` during LSTM fitting unless there is strong evidence current windows are IID-like

## Priority D: Add feature scaling for LSTM only
Apply scaling to LSTM input features using train-only fit.
Conservative default:
- `StandardScaler` on features
- optionally target scaling only if needed, but avoid unless carefully inverted at evaluation

Do not scale using full data before split.

## Priority E: Small XGB tuning
Try one-at-a-time changes such as:
- `n_estimators`: 300 -> 500
- `max_depth`: 4 -> 3 or 5
- `learning_rate`: 0.05 -> 0.03 or 0.1
- add `colsample_bytree`: 0.8
- add `reg_lambda`: moderate default if absent
- if supported cleanly, use early stopping with a validation set that is not the final test set

Avoid large search grids.

# Experiment loop

For each iteration:

1. Run baseline `train.py` unchanged
2. Record all reported metrics for XGB and LSTM
3. Pick one small hypothesis
4. Edit `train.py`
5. Re-run
6. Compare test RMSE / MAE / R² against baseline
7. Keep only if improvement is clear and evaluation remains fair
8. Update experiment log in comments or a lightweight text block if appropriate
9. Move to next small hypothesis

# Recommended experiment sequence

## Experiment 0: Baseline measurement
Goal:
- establish current SSH prediction accuracy

Actions:
- ensure metrics are printed for XGB, LSTM, and any ensemble
- include naive baseline(s)

Keep:
- only metric/reporting additions, no training changes yet

## Experiment 1: Time-aware LSTM validation
Hypothesis:
- current `validation_split=0.1` may be suboptimal or leaky for sequential data

Actions:
- replace `validation_split=0.1` with an explicit chronological validation split from the training sequences
- set `shuffle=False` in `model.fit(...)`
- preserve early stopping

Measure:
- test RMSE/MAE/R² for LSTM

Keep if:
- LSTM test metrics improve or become more stable/trustworthy

## Experiment 2: LSTM feature scaling
Hypothesis:
- mixed feature scales hurt LSTM optimization

Actions:
- fit scaler on train-only LSTM features
- transform train/val/test consistently
- keep architecture unchanged

Measure:
- LSTM metrics and training stability

## Experiment 3: XGB stronger defaults
Hypothesis:
- XGB can improve with a slightly better bias/variance tradeoff

Try one of:
- `max_depth=3` with `n_estimators=500`
- `max_depth=5` keeping learning rate low
- add `colsample_bytree=0.8`
- add `min_child_weight` modestly if overfitting suspected

Run one change at a time.

## Experiment 4: Feature ablation for XGB and/or LSTM
Hypothesis:
- some columns may add noise

Use the provided raw columns carefully:
- test all features vs dropping one suspicious feature at a time
- especially compare:
  - with vs without `time`
  - with vs without `latitude`,`longitude`
  - with vs without `ice_missing`
  - ensure `adt` remains included unless evidence strongly says otherwise

Because `adt` is likely highly informative for SSH, treat it as a core feature.

## Experiment 5: Persistence baseline / residual sanity check
Hypothesis:
- user wants “accurately predicts SSH levels”; absolute metrics alone may hide underperformance against simple temporal persistence

If sequence context allows:
- compare LSTM/XGB to previous-target baseline
- if model barely beats persistence, focus on improving temporal handling rather than enlarging the network

# Specific implementation guidance

## Metrics code
Use sklearn metrics if available:
- `mean_squared_error`
- `mean_absolute_error`
- `r2_score`

Print concise blocks like:
- `XGB Test RMSE: ...`
- `XGB Test MAE: ...`
- `XGB Test R2: ...`
- same for LSTM

## Baseline checks
Implement at least:
- mean baseline: predict `train["target"].mean()` for all test rows
If sequence indexing makes it easy, also:
- persistence baseline: predict previous observed target

## LSTM sequence handling
Be careful with indexing:
- do not create windows that cross train/test boundary if the split is supposed to simulate forecasting
- if current code builds `all_X` and `all_y` from all `df`, inspect `make_sequences(...)` and ensure train/test indices are partitioned correctly
- if unclear, prefer the conservative approach: build train sequences only from train rows and test sequences only from test rows

## Feature scaling
For LSTM:
- scale features using train-only stats
- preserve sequence shape after scaling
For XGB:
- scaling is usually unnecessary; avoid unless a very specific reason emerges

## Reproducibility
Set / preserve seeds where possible:
- numpy
- tensorflow / keras
- xgboost random state already visible

Do not spend much time on perfect determinism, but avoid accidental variability.

# Guardrails / anti-patterns

Do not:
- rewrite the whole pipeline
- introduce cross-validation that breaks sequence order
- tune directly on the final test set repeatedly without noting that this weakens trust
- leak future information into sequence windows
- fit scalers on full data before splitting
- make multiple major changes in one experiment
- optimize only training loss while ignoring test metrics
- remove existing outputs relied on by surrounding harness logic

Be cautious with:
- `validation_split` in Keras on temporal data
- any code that uses `df` globally before splitting
- any hidden ensemble logic not shown in the excerpt

# Conservative policy for ambiguity

If a detail is unclear from `train.py`:
- preserve current behavior
- add minimal instrumentation first
- choose the least invasive change that improves evaluation quality
- prefer improving validation/preprocessing over changing model families

If there is any conflict between better methodology and preserving functionality:
- preserve existing outputs and interfaces
- add new metrics/reporting alongside old behavior rather than replacing it

# Suggested logging format

Use a compact experiment note block in comments near the top of `train.py` or printed to stdout:

Experiment log:
- EXP00 baseline metrics only
  - XGB: RMSE=?, MAE=?, R2=?
  - LSTM: RMSE=?, MAE=?, R2=?
  - Baseline mean: RMSE=?, ...
  - Decision: baseline
- EXP01 chronological val split + shuffle=False for LSTM
  - Result: ...
  - Decision: keep/revert
- EXP02 LSTM StandardScaler on train-only features
  - Result: ...
  - Decision: keep/revert

# Definition of “accurately predicts SSH levels”

Interpret this pragmatically as:
- low RMSE/MAE on held-out `target`
- strong R²
- prediction distribution close to target distribution
- beats naive baselines
- no obvious bias or variance collapse

If possible, print a short final verdict such as:
- “Model beats mean baseline by X% RMSE”
- “Model underestimates variability”
- “LSTM/XGB is currently the best SSH predictor”

# Immediate first actions

1. Add explicit RMSE/MAE/R² reporting for XGB and LSTM on test
2. Add mean baseline, and persistence baseline if easy
3. Inspect whether LSTM validation is time-aware; if not, fix that first
4. Try train-only feature scaling for LSTM
5. Then try one small XGB hyperparameter improvement
6. Keep only changes that improve held-out SSH metrics fairly