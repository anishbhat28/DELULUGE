import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from pathlib import Path
import joblib

DATA_DIR = Path(__file__).parent

# ── 1. Load sea level data ──────────────────────────────────────────────────
gmsl_raw = pd.read_csv(DATA_DIR / 'gmsl-satelliterecord-copy.csv', skiprows=1)
gmsl_raw.columns = ['decimal_year', 'sea_level_mm', 'c', 'd']
gmsl_raw = gmsl_raw[['decimal_year', 'sea_level_mm']].dropna()
gmsl_raw['year'] = gmsl_raw['decimal_year'].astype(int)

# Annual mean sea level per year
gmsl = gmsl_raw.groupby('year')['sea_level_mm'].mean().reset_index()
gmsl.columns = ['year', 'sea_level_mm']

# ── 2. Load sea ice extent data ─────────────────────────────────────────────
ice_raw = pd.read_csv(DATA_DIR / 'Sea_Ice_Index_Monthly_Data_by_Year_G02135_v4.0.csv', header=0)
ice_raw.columns = ['year','Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec','_','Annual']
ice_raw = ice_raw[['year','Jan','Feb','Mar','Apr','May','Jun',
                   'Jul','Aug','Sep','Oct','Nov','Dec','Annual']]
ice_raw['year'] = pd.to_numeric(ice_raw['year'], errors='coerce')
ice_raw = ice_raw.dropna(subset=['year'])
ice_raw['year'] = ice_raw['year'].astype(int)

months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
for m in months:
    ice_raw[m] = pd.to_numeric(ice_raw[m], errors='coerce')
ice_raw['Annual'] = pd.to_numeric(ice_raw['Annual'], errors='coerce')

# ── 3. Engineer features ────────────────────────────────────────────────────
ice = ice_raw.copy()
ice['ice_annual_mean'] = ice[months].mean(axis=1)
ice['ice_sep_min'] = ice['Sep']                      # September = annual minimum
ice['ice_winter_mean'] = ice[['Jan','Feb','Mar']].mean(axis=1)
ice['ice_summer_mean'] = ice[['Jul','Aug','Sep']].mean(axis=1)
ice['ice_yoy_change'] = ice['ice_annual_mean'].diff()  # year-over-year change

# ── 4. Merge on overlapping years (1993–2022) ───────────────────────────────
df = pd.merge(ice, gmsl, on='year')
df = df[(df['year'] >= 1993) & (df['year'] <= 2022)].dropna()

print(f"Training on {len(df)} years ({df['year'].min()}–{df['year'].max()})")
print(df[['year','ice_sep_min','ice_annual_mean','sea_level_mm']].to_string(index=False))

# ── 5. Train model ──────────────────────────────────────────────────────────
features = ['ice_annual_mean', 'ice_sep_min', 'ice_winter_mean',
            'ice_summer_mean', 'ice_yoy_change']

X = df[features]
y = df['sea_level_mm']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

y_range = y.max() - y.min()

for name, model in [('Linear Regression', lr), ('Random Forest', rf)]:
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    nrmse = rmse / y_range
    print(f"\n{name}")
    print(f"  R²:    {r2_score(y_test, preds):.3f}")
    print(f"  RMSE:  {rmse:.2f} mm")
    print(f"  NRMSE: {nrmse:.4f}  ({nrmse*100:.2f}% of range)")

# ── 6. Predict over full timeline ──────────────────────────────────────────
df = df.sort_values('year').reset_index(drop=True)
X_all = df[features]
y_all = df['sea_level_mm']

lr_preds_all = lr.predict(X_all)
rf_preds_all = rf.predict(X_all)

# ── 7. Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 11))

# Plot 1: Sea ice + sea level trend
ax1 = axes[0, 0]
ax1b = ax1.twinx()
ax1.plot(df['year'], df['ice_sep_min'], color='steelblue', marker='o', markersize=4, label='Sep Ice Extent')
ax1b.plot(df['year'], df['sea_level_mm'], color='coral', marker='s', markersize=4, label='Sea Level Rise')
ax1.set_xlabel('Year')
ax1.set_ylabel('Ice Extent (million km²)', color='steelblue')
ax1b.set_ylabel('Sea Level Rise (mm)', color='coral')
ax1.set_title('Sea Ice Decline vs Sea Level Rise (1993–2022)')
ax1.grid(alpha=0.3)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1b.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# Plot 2: ML model predictions vs actual
ax2 = axes[0, 1]
ax2.plot(df['year'], y_all.values, color='black', linewidth=2, label='Actual', zorder=3)
ax2.plot(df['year'], lr_preds_all, color='dodgerblue', linestyle='--', marker='o', markersize=3, label='Linear Regression')
ax2.plot(df['year'], rf_preds_all, color='tomato', linestyle='--', marker='s', markersize=3, label='Random Forest')
ax2.set_title('ML Model Predictions vs Actual Sea Level')
ax2.set_xlabel('Year')
ax2.set_ylabel('Sea Level Rise (mm)')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Predicted vs Actual scatter (test set)
ax3 = axes[1, 0]
lr_test = lr.predict(X_test)
rf_test = rf.predict(X_test)
ax3.scatter(y_test, lr_test, color='dodgerblue', label='Linear Regression', alpha=0.8, zorder=3)
ax3.scatter(y_test, rf_test, color='tomato', label='Random Forest', alpha=0.8, zorder=3)
min_val = min(y_test.min(), lr_test.min(), rf_test.min())
max_val = max(y_test.max(), lr_test.max(), rf_test.max())
ax3.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
ax3.set_title('Predicted vs Actual (Test Set)')
ax3.set_xlabel('Actual Sea Level Rise (mm)')
ax3.set_ylabel('Predicted Sea Level Rise (mm)')
ax3.legend()
ax3.grid(alpha=0.3)

# Plot 4: Feature importance (Random Forest)
ax4 = axes[1, 1]
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)
ax4.barh([features[i] for i in sorted_idx], importances[sorted_idx], color='tomato', alpha=0.8)
ax4.set_title('Random Forest Feature Importance')
ax4.set_xlabel('Importance Score')
ax4.grid(alpha=0.3, axis='x')

plt.suptitle('Sea Ice → Sea Level Rise Prediction', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(DATA_DIR / 'climate_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved to climate_analysis.png")

# ── 8. Save models ───────────────────────────────────────────────────────────
joblib.dump(lr, DATA_DIR / 'linear_regression_model.pkl')
joblib.dump(rf, DATA_DIR / 'random_forest_model.pkl')
print("Models saved to linear_regression_model.pkl and random_forest_model.pkl")