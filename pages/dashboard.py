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
