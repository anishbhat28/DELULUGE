import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
from tensorflow import keras

HORIZON = 30
WINDOW = 60
LAG_DAYS = [7, 14, 30]

# ── Load & filter 1993-1995 ───────────────────────────────────────────────────
df = pd.read_csv("gulf_mexico_1993_1997.csv")
df["date"] = pd.to_datetime("1950-01-01") + pd.to_timedelta(df["time"], unit="D")
df = df[df["date"].dt.year <= 1995].reset_index(drop=True)

# Seasonality encoding
df["doy_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365)
df["doy_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365)

# ── Lag features (for XGBoost) ────────────────────────────────────────────────
for lag in LAG_DAYS:
    df[f"adt_lag_{lag}"] = df["adt"].shift(lag)
    df[f"ice_lag_{lag}"] = df["ice_extent"].shift(lag)

df["target"] = df["adt"].shift(-HORIZON)
df = df.dropna().reset_index(drop=True)

# ── Train/test split: 1993-1994 train, 1995 test ─────────────────────────────
train = df[df["date"].dt.year <= 1994].reset_index(drop=True)
test  = df[df["date"].dt.year == 1995].reset_index(drop=True)

XGB_FEATURES = (
    [f"adt_lag_{l}" for l in LAG_DAYS] +
    [f"ice_lag_{l}" for l in LAG_DAYS] +
    ["doy_sin", "doy_cos"]
)

X_train_xgb, y_train = train[XGB_FEATURES].values, train["target"].values
X_test_xgb,  y_test  = test[XGB_FEATURES].values,  test["target"].values

# ── XGBoost ──────────────────────────────────────────────────────────────────
xgb_model = xgb.XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                               subsample=0.8, random_state=42)
xgb_model.fit(X_train_xgb, y_train, eval_set=[(X_test_xgb, y_test)], verbose=False)
xgb_preds = xgb_model.predict(X_test_xgb)

xgb_mae  = mean_absolute_error(y_test, xgb_preds)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_preds))
print(f"XGBoost  — MAE: {xgb_mae:.4f}  RMSE: {xgb_rmse:.4f}")

# ── LSTM data prep ────────────────────────────────────────────────────────────
LSTM_FEATURES = ["adt", "ice_extent", "doy_sin", "doy_cos"]

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

all_X = df[LSTM_FEATURES].values
all_y = df["target"].values.reshape(-1, 1)

train_idx = df[df["date"].dt.year <= 1994].index
test_idx  = df[df["date"].dt.year == 1995].index

scaler_X.fit(all_X[train_idx])
scaler_y.fit(all_y[train_idx])

scaled_X = scaler_X.transform(all_X)
scaled_y = scaler_y.transform(all_y).flatten()

def make_sequences(X, y, indices, window):
    seqs, targets = [], []
    for i in indices:
        if i < window:
            continue
        seqs.append(X[i - window:i])
        targets.append(y[i])
    return np.array(seqs), np.array(targets)

X_train_lstm, y_train_lstm = make_sequences(scaled_X, scaled_y, train_idx, WINDOW)
X_test_lstm,  y_test_lstm  = make_sequences(scaled_X, scaled_y, test_idx,  WINDOW)

# ── LSTM model ────────────────────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.LSTM(64, input_shape=(WINDOW, len(LSTM_FEATURES)), return_sequences=True),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(32),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1),
])
model.compile(optimizer="adam", loss="mse")

early_stop = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
model.fit(X_train_lstm, y_train_lstm, epochs=200, batch_size=32,
          validation_split=0.1, callbacks=[early_stop], verbose=0)

lstm_preds_scaled = model.predict(X_test_lstm).flatten()
lstm_preds = scaler_y.inverse_transform(lstm_preds_scaled.reshape(-1, 1)).flatten()
y_test_lstm_orig = scaler_y.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()

lstm_mae  = mean_absolute_error(y_test_lstm_orig, lstm_preds)
lstm_rmse = np.sqrt(mean_squared_error(y_test_lstm_orig, lstm_preds))
print(f"LSTM     — MAE: {lstm_mae:.4f}  RMSE: {lstm_rmse:.4f}")




# --- auto-injected predictions.csv writer (contract shim) ---
try:
    import pandas as _pd
    import numpy as _np
    from pathlib import Path as _Path
    _g = globals()

    def _is_1d_num_array(v):
        try:
            a = _np.asarray(v)
            return a.ndim == 1 and a.size > 0 and _np.issubdtype(a.dtype, _np.number)
        except Exception:
            return False

    _arrays = {k: _np.asarray(v) for k, v in list(_g.items())
               if not k.startswith('_') and _is_1d_num_array(v)}
    _target_keys = [k for k in _arrays if any(s in k.lower() for s in ('y_test', 'target', 'y_true', 'y_val'))]
    _pred_keys = [k for k in _arrays if 'pred' in k.lower() and 'lag' not in k.lower()]

    _t_arr, _p_arr, _picked_t, _picked_p = None, None, None, None
    for _tk in _target_keys:
        for _pk in _pred_keys:
            if len(_arrays[_tk]) == len(_arrays[_pk]):
                _t_arr, _p_arr, _picked_t, _picked_p = _arrays[_tk], _arrays[_pk], _tk, _pk
                break
        if _t_arr is not None:
            break

    if _t_arr is not None:
        _out = _pd.DataFrame({'target': _t_arr, 'prediction': _p_arr})
        # Pull matching feature/context columns from a test DataFrame if present
        for _candidate_name in ('test', 'test_df', 'df_test', 'X_test_df', 'val', 'val_df'):
            _cand = _g.get(_candidate_name)
            if isinstance(_cand, _pd.DataFrame) and len(_cand) == len(_t_arr):
                for _c in _cand.columns:
                    if _c in _out.columns:
                        continue
                    try:
                        _col_arr = _cand[_c].to_numpy()
                        if _np.issubdtype(_col_arr.dtype, _np.number):
                            _out[_c] = _col_arr
                    except Exception:
                        pass
                break
        _out.to_csv(_Path(__file__).parent / 'predictions.csv', index=False)
        print(f"auto-injected: wrote predictions.csv ({len(_out)} rows, cols: {list(_out.columns)}) from {_picked_t}/{_picked_p}")
    else:
        print(f"auto-inject skipped: no matching target/prediction arrays (targets={_target_keys}, preds={_pred_keys})")
except Exception as _e:
    print(f"auto-inject skipped: {_e}")
