# Mission

Improve `train.py` to better predict ocean SSH / sea level (`sea_level_mm`) using short, evidence-driven experiments.

The current code already trains at least:
- `LinearRegression`
- `RandomForest` (inferred from `rf` usage)

and produces:
- predictions over all years
- test-set scatter plot
- serialized linear regression model via `joblib.dump`

Your job is to make small, justified improvements that increase predictive quality without breaking existing functionality.

# What to edit

Edit only:
- `train.py`

You may:
- add imports
- add helper functions inside `train.py`
- adjust feature selection, splitting, model definitions, preprocessing, metrics, plotting labels, and model saving
- add lightweight logging/printing of metrics

# What not to edit

Do not edit unless absolutely required by the script:
- raw data files
- directory structure
- unrelated files
- plotting/output behavior in a way that removes existing outputs

Preserve:
- ability to train and evaluate models
- existing visualizations if possible
- model serialization behavior, or replace it with an equally clear and correct save path only if needed

# Problem understanding

Target:
- `sea_level_mm`

Available columns:
- `year`
- monthly values: `Jan` … `Dec`
- `Annual`
- `sea_level_mm`

Most likely task:
- supervised regression from year and temperature-like monthly/annual climate features to sea level / SSH

Important data characteristics inferred:
- one row per year
- likely small dataset (starts at 1993 in sample; probably only a few decades total)
- temporal order matters
- leakage risk if random train/test split is used
- `Annual` is probably derived from monthly columns, so using both monthly features and `Annual` may introduce redundancy/collinearity
- `year` is likely strongly informative due to sea-level trend

# Primary optimization target

Optimize test-set regression quality for `sea_level_mm`.

Use, print, and compare at minimum:
1. `R^2`
2. `RMSE`
3. `MAE`

If only one metric is needed for deciding changes, prioritize:
1. `RMSE` as primary decision metric
2. `R^2` as secondary tie-breaker

Reason:
- sea level is continuous and trend-like
- RMSE is interpretable in target units (`mm`)
- small dataset means R² may be unstable

# Most likely weaknesses in the current code

These are the highest-probability improvement targets based on the excerpts:

1. Temporal split may be missing or weak
   - The dataframe is sorted by `year`, but it is unclear whether train/test split respects chronology.
   - For forecasting-like yearly sea-level prediction, train on past and test on future years only.

2. Linear model may be underpowered but is a strong baseline
   - `LinearRegression` should likely remain as a baseline.
   - Regularized linear models may outperform plain OLS on small correlated feature sets.

3. Random forest may overfit badly on a tiny tabular time series
   - Small yearly datasets + many correlated features often make RF unstable.
   - Tune conservatively or compare against simpler models first.

4. Feature redundancy
   - `Annual` duplicates information from monthly columns.
   - `year` may dominate.
   - Need explicit, controlled comparison of feature subsets.

5. Missing evaluation discipline
   - Existing script likely plots predictions, but may not print enough metrics for fair model comparison.

# Constraints and conservative assumptions

Because the full script is not shown, follow these conservative policies:

- Assume runtime should stay fast: target each experiment to finish in well under 1–2 minutes.
- Assume CPU-only compatibility.
- Do not add heavy dependencies unless they are already available in the environment.
- Prefer `scikit-learn` models and utilities.
- Keep deterministic behavior by setting `random_state` anywhere applicable.
- Preserve existing plots and save behavior where feasible.
- If a train/test split already exists, do not silently change it without clearly logging the new split policy.

# Fair comparison rules

All experiments must be compared fairly.

For every run:
- use the same dataset
- use the same split policy
- use the same metrics
- use fixed random seeds
- compare against the current baseline in the same run if possible

If you change the split strategy:
- first establish a new baseline under the new split
- only compare later experiments against that new baseline

# Recommended evaluation protocol

## Preferred split
Use chronological split:
- sort by `year`
- train on earlier years
- test on later years

Conservative default:
- last 20–30% of years as test set
- do not shuffle

If the script currently uses `train_test_split(..., shuffle=True)`, replace it with a time-aware split unless that would break explicit user-facing expectations.

## Optional stronger validation
If feasible with small code changes, add a second evaluation:
- rolling or expanding-window validation on the training set only

Examples:
- `TimeSeriesSplit`
- manual expanding-window folds

Use this only for model selection; keep the final reported score on the held-out final test period.

# Minimum metrics to print for each model

For train and test if easy:
- `R2`
- `RMSE`
- `MAE`

At minimum on test set:
- `R2`
- `RMSE`
- `MAE`

Suggested print format:
- `ModelName | Test RMSE: X.XXX | Test MAE: X.XXX | Test R2: X.XXX`

If train metrics are printed too:
- this helps diagnose overfitting, especially for RF

# Safe modification priorities

Apply changes in this order.

## Priority 1: Make evaluation robust and explicit
Before tuning models, ensure:
- chronological split
- consistent metrics
- deterministic seeds
- clear printed comparison table

This is the highest-value improvement if missing.

## Priority 2: Add strong small-data baselines
Try these with minimal code:
- `Ridge`
- `Lasso`
- `ElasticNet`

Use scaling for these models:
- `StandardScaler` in a `Pipeline`

Reason:
- monthly + annual features are correlated
- regularization often helps more than RF on tiny tabular datasets

## Priority 3: Controlled feature subset experiments
Run small ablations:

1. `year` only
2. `year + Annual`
3. `year + monthly columns`
4. `monthly columns only`
5. `year + monthly + Annual`

Reason:
- `year` may capture trend
- `Annual` may summarize months well
- monthly data may add seasonal climate information
- comparing these tells you whether SSH is mostly trend-driven or also tied to intra-annual climate structure

## Priority 4: Conservative random forest tuning
If RF is already present, tune only a few parameters:
- reduce overfitting
- keep runtime low

Try:
- `n_estimators`: 100, 300
- `max_depth`: 3, 5, None
- `min_samples_leaf`: 1, 2, 4
- `max_features`: `sqrt`, `0.5`, `1.0`

Do not perform large grid searches.
Use 3–6 targeted configurations max.

## Priority 5: Simple polynomial trend for linear baseline
If `year` is highly predictive, try:
- polynomial features on `year` only or `year + Annual`
- degree 2 only
- use `Pipeline([PolynomialFeatures, StandardScaler, Ridge])` if appropriate

Do not go beyond degree 2 unless clearly justified.

# Feature handling guidance

## Candidate feature groups

Define explicit feature sets:
- `F1 = ['year']`
- `F2 = ['year', 'Annual']`
- `F3 = ['Jan', 'Feb', ..., 'Dec']`
- `F4 = ['year'] + months`
- `F5 = months + ['Annual']`
- `F6 = ['year'] + months + ['Annual']`

## Redundancy caution
`Annual` is likely derived from monthly values.
Therefore:
- treat inclusion of `Annual` as an experiment, not a default necessity
- do not assume `Annual` always helps

## Scaling policy
Use scaling for:
- `LinearRegression` optional
- `Ridge`, `Lasso`, `ElasticNet` yes
- polynomial models yes

Do not scale for:
- `RandomForest`

# Concrete first experiments

Run these in order and stop when improvements are exhausted.

## Experiment 0: Establish trustworthy baseline
Goal:
- keep current models
- ensure chronological split
- print RMSE/MAE/R² for each model

Expected value:
- may reveal current performance is optimistic if random split was used

Keep if:
- evaluation becomes clearer and reproducible without breaking outputs

## Experiment 1: Add Ridge baseline
Add:
- `Pipeline([StandardScaler(), Ridge(alpha=1.0)])`

Compare against:
- `LinearRegression`
- existing `RandomForest`

Also try small alpha values if easy:
- `0.1`, `1.0`, `10.0`

Keep if:
- test RMSE improves meaningfully over linear regression
- or performance matches with lower train/test gap

## Experiment 2: Feature ablation with the best simple model
Using the best of `LinearRegression`/`Ridge`, compare:
- `year` only
- `year + Annual`
- `year + months`
- `year + months + Annual`

Keep if:
- one feature set gives clearly better test RMSE
- use the simplest feature set within ~1–2% of best RMSE

## Experiment 3: Regularized sparse model
Add:
- `Lasso` or `ElasticNet` with scaling

Try very small set of values:
- `Lasso(alpha=0.001, 0.01, 0.1)`
- or `ElasticNet(alpha=0.001/0.01/0.1, l1_ratio=0.2/0.5/0.8)` but keep combinations limited

Use only if easy.
Keep if:
- better test RMSE
- or similar RMSE with simpler effective feature use

## Experiment 4: Conservative RF tuning
If RF underperforms badly, either:
- leave it as a baseline
- or make one small tuning pass

Likely useful settings for tiny data:
- shallower trees
- larger `min_samples_leaf`

Keep only if:
- test RMSE improves over current RF and is competitive with linear models

## Experiment 5: Quadratic year trend
Test:
- `year` transformed to polynomial degree 2
- model with `Ridge`

Feature sets:
- `year` only
- `year + Annual`

Keep only if:
- test RMSE improves over linear year-based models
- no obvious severe overfit on test scatter or train/test gap

# Experiment workflow

For each experiment:

1. Read current `train.py`
2. Make one small change only
3. Run the script
4. Record:
   - feature set
   - split policy
   - models tested
   - metric values
   - whether plots still render/save
5. Compare against baseline
6. Keep the change only if justified
7. Otherwise revert before the next experiment

Do not stack multiple unvalidated changes.

# Decision rules: better vs worse

A change is better if:
- test RMSE decreases by a meaningful amount, or
- test RMSE is tied and R²/MAE improve slightly, or
- performance is essentially tied but the model is simpler and more robust

A change is worse if:
- test RMSE increases
- performance becomes unstable across reruns due to missing seeds
- code becomes significantly more complex without clear metric gains
- plots / saved artifacts break

Conservative threshold:
- prefer keeping only changes with at least ~2–5% RMSE improvement, unless the change also fixes a clear methodological flaw (such as leakage or random split on time data)

# Revert policy

Immediately revert a change if any of the following happen:
- script fails
- metrics are not produced
- existing outputs are broken without replacement
- train/test split becomes ambiguous
- test RMSE is worse with no clear methodological gain
- complexity grows significantly for marginal benefit

If a methodological fix causes metrics to worsen but makes evaluation more valid:
- keep that fix
- treat the new result as the new baseline

Example:
- switching from random split to chronological split may reduce apparent scores; this is expected and should usually be kept

# Logging format

Add lightweight, human-readable experiment logs to stdout and/or comments near the top of the file if appropriate.

For each run, log:

- experiment ID
- date/time if easy
- split description
- feature set
- model configuration
- metrics

Suggested stdout block:

`EXP: E2_ridge_year_annual`
`Split: chronological, last 20% test`
`Features: year, Annual`
`Model: Pipeline(StandardScaler + Ridge(alpha=1.0))`
`Test RMSE: ...`
`Test MAE: ...`
`Test R2: ...`

If multiple models are evaluated in one run, print a compact table.

# Guardrails / anti-patterns

Avoid:
- random shuffling for final time-based evaluation
- giant hyperparameter sweeps
- adding deep learning or heavy frameworks
- replacing everything at once
- using future information in training
- keeping both monthly and annual features without testing whether redundancy hurts
- evaluating only on plots without numeric metrics
- introducing nondeterminism by forgetting `random_state`

Be careful with:
- very small datasets: a single outlier year can dominate
- overfitting RF on tiny annual records
- interpreting a tiny R² difference as meaningful

# Implementation suggestions

If needed, organize code into small helper blocks:
- feature set dictionary
- metric computation helper
- model dictionary
- chronological split helper

But keep changes incremental; do not do a full refactor first.

Suggested metrics helper:
- compute `rmse`, `mae`, `r2`

Suggested model set for early iterations:
- `LinearRegression()`
- `Pipeline([StandardScaler(), Ridge(alpha=1.0)])`
- existing `RandomForestRegressor(random_state=42, ...)`

# Preferred final state

A good final `train.py` should:
- sort by year
- use a chronological train/test split
- explicitly predict `sea_level_mm`
- compare a few strong tabular baselines
- print RMSE/MAE/R²
- keep existing plots
- save at least the best simple model or preserve existing serialization
- remain short and readable

# If something is ambiguous

Use this conservative policy:
- preserve current behavior unless there is a clear methodological issue
- prioritize valid time-aware evaluation over optimistic scores
- prefer simpler models on this small dataset
- make one change at a time and keep only changes supported by test metrics

# Immediate next steps

1. Inspect how `X` and `y` are currently built and how the split is done.
2. Ensure the target is `sea_level_mm`.
3. Ensure split is chronological, not shuffled.
4. Add printed `RMSE`, `MAE`, `R²` for current linear regression and random forest.
5. Add `Ridge` with scaling as the first new model.
6. Run feature ablations centered on `year`, `Annual`, and monthly columns.
7. Keep the simplest configuration with the best held-out test RMSE.