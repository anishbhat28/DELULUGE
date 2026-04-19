# Mission

Improve `train.py` to better predict ocean SSH / sea level (`sea_level_mm`) using short, high-signal experiments.

The current script appears to:
- fit at least `LinearRegression` and `RandomForest`
- split data into train/test
- plot predictions over `year`
- save a linear regression model with `joblib`

Your goal is to make small, justified changes that improve predictive quality while preserving the script’s current behavior and outputs as much as possible.

# What to edit

Edit only:
- `train.py`

Do not edit unless absolutely required for the experiment:
- data files
- file names / paths
- external configuration
- plotting behavior beyond what is needed to support better evaluation

Preserve:
- ability to run end-to-end
- model training and prediction plots
- model saving behavior unless there is a strong reason to extend it

# Problem understanding from the code and data

Likely target:
- `sea_level_mm`

Likely features:
- `year`
- monthly columns `Jan` ... `Dec`
- `Annual`

Important data characteristics:
- This is annual/tabular data indexed by `year`
- Data is sorted by `year` in the script
- Because SSH / sea level is time-ordered, random shuffling would risk leakage or unrealistically easy evaluation
- `Annual` is probably derived from monthly values, so using both monthly columns and `Annual` may add redundancy; this is acceptable for tree models but may affect linear models via collinearity
- `year` may be highly predictive because sea level often has a long-term trend; include/exclude it deliberately and measure impact rather than assuming

# Likely optimization targets in this codebase

Based on the excerpt, the highest-value targets are:

1. Evaluation quality
   - The script likely compares models visually, but improvement decisions should be driven by explicit numeric metrics on a held-out time-ordered test set.

2. Train/test split correctness
   - Since the script sorts by year, preserve temporal ordering in train/test split.
   - Avoid random split if present elsewhere in the file.

3. Baseline model quality
   - `LinearRegression` is a useful baseline but probably too simple.
   - `RandomForest` is already present; its hyperparameters are likely default and worth light tuning.

4. Feature handling
   - Compare:
     - monthly features only
     - monthly + `year`
     - monthly + `Annual`
     - monthly + `Annual` + `year`
   - Redundant features should be tested, not assumed helpful.

5. Scaling and regularized linear models
   - If only plain linear regression is used, try `Ridge` and `Lasso` with scaling.
   - This is a small, safe extension with good signal on tabular regression.

# Primary metric to optimize

Optimize test-set regression quality for `sea_level_mm`.

Use these metrics, in this order:
1. `R^2` on test set: higher is better
2. `RMSE` on test set: lower is better
3. `MAE` on test set: lower is better

If the current script already computes another explicit numeric validation metric, preserve it and prioritize it. If not, add the three metrics above.

Decision rule:
- A change is better only if it improves the primary metric (`R^2`) and does not meaningfully worsen RMSE/MAE.
- If `R^2` is unstable due to small test size, prefer lower RMSE and MAE with similar `R^2`.

# Fair comparison rules

To compare runs fairly:
- Keep the same train/test split across experiments.
- Preserve temporal ordering: train on earlier years, test on later years.
- Use fixed `random_state` for any stochastic model.
- Do not compare a model trained on a different split unless the experiment is specifically about split strategy.
- Change one main thing at a time.

If there is currently a random split in `train.py`, replace it with a deterministic chronological split and use that consistently for all further experiments.

Recommended conservative split policy:
- sort by `year`
- use the earliest ~80% of years for training and latest ~20% for test
- no shuffling

# Minimum evaluation to add if missing

Ensure `train.py` prints a compact summary table for each model:
- model name
- feature set name
- train size / test size
- test R²
- test RMSE
- test MAE

If easy, also print train R² to spot overfitting, especially for Random Forest.

# Safe modification priorities

Prioritize these in order:

## 1) Improve evaluation clarity without changing modeling much
Add:
- explicit `R^2`, `RMSE`, `MAE`
- fixed chronological split
- fixed `random_state` for Random Forest
- clean comparison printout

This is the safest first step and gives a reliable baseline.

## 2) Light hyperparameter tuning for Random Forest
Try small changes only. Most likely useful parameters:
- `n_estimators`
- `max_depth`
- `min_samples_leaf`
- `min_samples_split`
- `max_features`

Suggested values to test in short iterations:
- `n_estimators`: 100, 300, 500
- `max_depth`: None, 3, 5, 8
- `min_samples_leaf`: 1, 2, 4
- `min_samples_split`: 2, 4, 8

Use a tiny manual grid, not a large expensive search.

## 3) Add regularized linear baselines
Test:
- `Ridge`
- optionally `Lasso`

Use:
- `Pipeline([StandardScaler(), Ridge(...)])`
- small alpha grid: `0.01, 0.1, 1.0, 10.0, 100.0`

This is especially relevant because features are continuous and potentially collinear (`Annual` vs monthly values).

## 4) Feature-set ablations
Try a few compact feature sets:
- A: `Jan`-`Dec`
- B: `Jan`-`Dec` + `Annual`
- C: `Jan`-`Dec` + `year`
- D: `Jan`-`Dec` + `Annual` + `year`

Optional additional ablations if quick:
- E: `Annual` + `year`
- F: `year` only

This helps identify whether the model is mostly using trend (`year`) or actual ocean/climate monthly features.

## 5) Keep plotting aligned with best model(s)
Once a better model is found, update plots to include it clearly. Do not remove existing useful visualizations.

# Conservative policies for ambiguities

Because only excerpts are visible, follow these conservative rules:

- If the script already has a split, do not replace it unless it is random or leaks future data.
- If there is already a metric printed, keep it and add `RMSE`/`MAE` only if absent.
- If there is already a saved model artifact, do not rename or remove it; add additional saved artifacts only if necessary.
- If there are missing values, handle them minimally and explicitly (e.g. drop rows with missing target; impute features only if required and keep it simple).
- Do not introduce heavy dependencies or large refactors.
- Do not build a neural network for this dataset unless all simpler baselines clearly underperform and the code structure makes it trivial.

# Experiment loop

For each experiment:

1. Read current `train.py`
2. Make one focused change
3. Run the script
4. Record:
   - exact code change
   - feature set
   - split details
   - metrics for each model
   - whether plots/model saving still work
5. Compare against the current best run
6. Keep the change only if it clearly helps
7. Revert immediately if:
   - metrics worsen
   - outputs break
   - code becomes significantly more complex without benefit

# Revert rules

Revert a change if any of the following happen:
- test `R^2` drops versus the best known run
- RMSE or MAE gets clearly worse with no compensating gain
- the script no longer runs end-to-end
- plots or saved model behavior break without a strong reason
- the change adds complexity disproportionate to the performance gain

If a change has mixed results:
- prefer the simpler version
- prefer the model with better test metrics on the chronological holdout
- do not keep speculative complexity

# Logging format

Append a short experiment note after each run in a consistent format, e.g.:

## Experiment N: <short title>
- Date/time: <timestamp>
- Change:
  - <1-3 bullets>
- Split:
  - chronological by year, train first 80%, test last 20%
- Features:
  - <feature set>
- Models:
  - <models tested>
- Results:
  - LinearRegression: R2=..., RMSE=..., MAE=...
  - RandomForest: R2=..., RMSE=..., MAE=...
  - Ridge(alpha=...): R2=..., RMSE=..., MAE=...
- Outcome:
  - kept / reverted
- Reason:
  - <why>

If no external log file exists, it is fine to keep this as comments in your working notes, but ensure the decision process is explicit.

# Guardrails / anti-patterns

Avoid:
- random train/test split on time series-like data
- evaluating only on train data or all data
- making many hyperparameter changes at once
- giant grid searches
- deleting the baseline models
- silently changing the target or feature definitions
- relying only on plots without numeric metrics
- tuning on the test set repeatedly without caution

If you add internal model selection:
- use only the training portion for tuning
- keep the final test period untouched for final comparison

Given likely small dataset size, prefer simple deterministic methods over data-hungry models.

# First experiment ideas tailored to this train.py

## Experiment 1: Add proper metrics and enforce chronological split
Goal:
- Establish a trustworthy baseline.

Changes:
- Ensure data sorted by `year`
- Create deterministic chronological train/test split
- Print `R^2`, `RMSE`, `MAE` for existing `LinearRegression` and `RandomForest`
- Set `random_state` for Random Forest if missing

Keep if:
- script remains functional
- metrics are clearly printed
- split is leakage-safe

## Experiment 2: Feature ablation for `year`
Goal:
- Determine whether long-term trend alone drives performance.

Compare at least:
- monthly + Annual
- monthly + Annual + year

If easy, also test:
- year only
- monthly only

Keep:
- feature set that gives best chronological test metrics
- preserve interpretability in logs

## Experiment 3: Add Ridge baseline
Goal:
- Improve over plain linear regression under collinearity.

Implementation:
- add `Pipeline(StandardScaler(), Ridge(alpha=...))`
- test small alpha grid
- compare against current linear regression and Random Forest

Keep if:
- Ridge beats LinearRegression on test metrics
- code impact remains small

## Experiment 4: Light Random Forest tuning
Goal:
- Improve nonlinear baseline with minimal search.

Suggested small search:
- `n_estimators`: 100 vs 300
- `max_depth`: None vs 5
- `min_samples_leaf`: 1 vs 2 vs 4

Do not run a large grid.
Keep only if the tuned RF materially improves test RMSE/MAE or R².

## Experiment 5: Overfitting check
Goal:
- See whether RF is memorizing.

Add:
- train R² and test R²

Interpretation:
- very high train R² with much worse test R² suggests overfitting
- if so, prefer shallower trees or larger `min_samples_leaf`

# Implementation notes likely useful in this codebase

- The target should remain `sea_level_mm`.
- Keep `year` available for plotting even if excluded from a feature set.
- If using multiple feature sets, name them explicitly in output.
- If saving a model, prefer saving the best-performing simple model in addition to preserving current save behavior.
- Because plots already compare actual vs predicted and prediction-vs-actual, maintain those diagnostics.

# Definition of success

A successful iteration:
- preserves end-to-end execution
- uses a fair chronological evaluation
- produces explicit test metrics
- improves held-out prediction of `sea_level_mm`
- keeps code understandable and compact

Focus on reliable small wins, not large rewrites.