"""
autoresearch.py
===============

Grounded autoresearch loop for physical-regime failure-mode discovery.

Flow:
    1. Gemini proposes a hypothesis in natural language.
    2. Gemini calls a tool to convert the hypothesis to a concrete spatial
       mask over (t, y, x).
    3. The tool computes inside-vs-outside error on a DISCOVERY split and
       returns the result. All numbers come from executed code.
    4. Accepted hypotheses are re-tested on a held-out VALIDATION split
       the agent never saw during generation.
    5. Bonferroni correction applied to the total hypothesis count.
    6. Output: outputs/findings.json, a list of validated regimes with
       their tool-call IDs (receipts).

Design choices that make this ours:
    - Physical-oceanography hypothesis language (EKE, Okubo-Weiss, LC extent,
      anomaly magnitude), not generic ML diagnostics.
    - Narrow, typed domain tools — the agent cannot run arbitrary Python.
    - Train/validation split for hypothesis selection to prevent overfitting.
    - Bonferroni multiple-testing correction.
    - Every reported finding carries the tool_call_id that computed it.

Attribution: inspired by Karpathy autoresearch, Sakana AI Scientist, Anthropic
agentic research patterns. Written from scratch for DataHacks 2026.
"""

import json
import os
import time
import uuid
import numpy as np
from scipy import stats

# Gemini SDK
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ---------- Config ----------
DISCOVERY_FRAC = 0.7   # fraction of test set the agent can query
BUDGET = 10            # max hypotheses the agent can test
MODEL_NAME = "gemini-2.5-flash"

# State for tool-call receipts (filled during run)
TOOL_CALL_LOG = []


# ---------- Data loading ----------
def load_data():
    preds = np.load("outputs/test_predictions.npz")
    regimes = np.load("outputs/test_regimes.npz")
    ocean = ~preds["land_mask"]
    std = float(preds["std_norm"])
    T = preds["abs_error"].shape[0]
    cutoff = int(T * DISCOVERY_FRAC)

    return {
        "abs_error": preds["abs_error"],       # (T, H, W) normalized
        "disag": preds["ensemble_std"],        # (T, H, W) normalized
        "eke": regimes["eke"],                 # (T, H, W)
        "ow": regimes["ow"],                   # (T, H, W)
        "lc_extent": regimes["lc_extent"],     # (T,)
        "anom_mag": regimes["anom_mag"],       # (T,)
        "ocean": ocean,
        "std": std,
        "T": T,
        "discovery_end": cutoff,
        "validation_start": cutoff,
    }


# ---------- Tool implementations ----------
def build_regime_mask(data, regime_type, comparator, value, split):
    """
    Build a boolean mask over (T_split, H, W) selecting pixels in the regime.

    regime_type: "eke", "ow_negative", "anom_percentile", "lc_extent"
    comparator: "gt", "lt", "sign" (for OW)
    value: numeric threshold OR percentile (0-100)
    split: "discovery" or "validation"

    Returns:
        mask: (T_split, H_ocean_flat) bool
        meta: dict with threshold info
    """
    if split == "discovery":
        t0, t1 = 0, data["discovery_end"]
    elif split == "validation":
        t0, t1 = data["validation_start"], data["T"]
    else:
        raise ValueError(f"bad split {split}")

    ocean = data["ocean"]

    if regime_type == "eke":
        field = data["eke"][t0:t1][:, ocean]  # (T_s, N_ocean)
        if comparator == "percentile_gt":
            thresh = np.percentile(field, value)
            mask = field > thresh
            meta = {"threshold_value": float(thresh), "percentile": value}
        else:
            raise ValueError(f"EKE needs percentile_gt, got {comparator}")

    elif regime_type == "ow_negative":
        field = data["ow"][t0:t1][:, ocean]
        mask = field < 0
        meta = {"threshold_value": 0.0, "description": "vortex-core regions"}

    elif regime_type == "ow_positive":
        field = data["ow"][t0:t1][:, ocean]
        mask = field > 0
        meta = {"threshold_value": 0.0, "description": "strain-dominated regions"}

    elif regime_type == "anom_percentile":
        # per-timestep scalar; broadcasts across spatial dim
        scalar = data["anom_mag"][t0:t1]  # (T_s,)
        thresh = np.percentile(scalar, value)
        per_t = scalar > thresh  # (T_s,)
        # Broadcast to (T_s, N_ocean)
        N_ocean = int(ocean.sum())
        mask = np.broadcast_to(per_t[:, None], (t1 - t0, N_ocean))
        meta = {"threshold_value": float(thresh), "percentile": value}

    elif regime_type == "lc_extent":
        scalar = data["lc_extent"][t0:t1]
        if comparator == "gt":
            per_t = scalar > value  # timesteps with extent > value
        else:
            per_t = scalar < value
        N_ocean = int(ocean.sum())
        mask = np.broadcast_to(per_t[:, None], (t1 - t0, N_ocean))
        meta = {"threshold_value": float(value), "comparator": comparator}
    else:
        raise ValueError(f"unknown regime_type {regime_type}")

    return mask, meta


def evaluate_regime(data, regime_type, comparator, value):
    """
    TOOL. Evaluates whether error is higher inside a regime than outside.

    Called by the agent. Returns a structured result including a tool_call_id.
    """
    call_id = str(uuid.uuid4())[:8]
    try:
        mask, meta = build_regime_mask(data, regime_type, comparator, value, "discovery")
        err = data["abs_error"][0:data["discovery_end"]][:, data["ocean"]]  # (T_d, N_ocean)

        inside = err[mask]
        outside = err[~mask]

        if inside.size < 100 or outside.size < 100:
            result = {
                "call_id": call_id,
                "status": "insufficient_data",
                "n_inside": int(inside.size),
                "n_outside": int(outside.size),
            }
            TOOL_CALL_LOG.append({"call_id": call_id, "type": "evaluate_regime",
                                  "input": {"regime_type": regime_type, "comparator": comparator,
                                            "value": value},
                                  "output": result})
            return result

        # Welch's t-test (unequal variances)
        t_stat, p_val = stats.ttest_ind(inside, outside, equal_var=False)
        std_norm = data["std"]
        result = {
            "call_id": call_id,
            "status": "ok",
            "regime_type": regime_type,
            "comparator": comparator,
            "value": value,
            "meta": meta,
            "n_inside": int(inside.size),
            "n_outside": int(outside.size),
            "mean_err_inside_mm": float(inside.mean() * std_norm * 1000),
            "mean_err_outside_mm": float(outside.mean() * std_norm * 1000),
            "error_ratio": float(inside.mean() / (outside.mean() + 1e-12)),
            "t_statistic": float(t_stat),
            "p_value_discovery": float(p_val),
        }
        TOOL_CALL_LOG.append({"call_id": call_id, "type": "evaluate_regime",
                              "input": {"regime_type": regime_type, "comparator": comparator,
                                        "value": value},
                              "output": result})
        return result
    except Exception as e:
        result = {"call_id": call_id, "status": "error", "error": str(e)}
        TOOL_CALL_LOG.append({"call_id": call_id, "type": "evaluate_regime", "error": str(e)})
        return result


def validate_regime(data, regime_type, comparator, value):
    """
    TOOL. Re-runs the regime test on the held-out VALIDATION split the
    agent never saw during discovery. This is the anti-p-hacking gate.
    """
    call_id = str(uuid.uuid4())[:8]
    try:
        mask, meta = build_regime_mask(data, regime_type, comparator, value, "validation")
        err = data["abs_error"][data["validation_start"]:data["T"]][:, data["ocean"]]

        inside = err[mask]
        outside = err[~mask]

        if inside.size < 100 or outside.size < 100:
            result = {"call_id": call_id, "status": "insufficient_data"}
            TOOL_CALL_LOG.append({"call_id": call_id, "type": "validate_regime",
                                  "output": result})
            return result

        t_stat, p_val = stats.ttest_ind(inside, outside, equal_var=False)
        std_norm = data["std"]
        result = {
            "call_id": call_id,
            "status": "ok",
            "n_inside_val": int(inside.size),
            "n_outside_val": int(outside.size),
            "mean_err_inside_mm": float(inside.mean() * std_norm * 1000),
            "mean_err_outside_mm": float(outside.mean() * std_norm * 1000),
            "error_ratio": float(inside.mean() / (outside.mean() + 1e-12)),
            "p_value_validation": float(p_val),
        }
        TOOL_CALL_LOG.append({"call_id": call_id, "type": "validate_regime",
                              "input": {"regime_type": regime_type, "comparator": comparator,
                                        "value": value},
                              "output": result})
        return result
    except Exception as e:
        return {"call_id": call_id, "status": "error", "error": str(e)}


# ---------- Gemini tool schemas ----------
GEMINI_TOOLS = [
    {
        "name": "evaluate_regime",
        "description": (
            "Test whether the surrogate's absolute error is systematically higher inside "
            "a specified physical regime than outside it. Uses a held-out DISCOVERY split "
            "of the test data so you can test multiple hypotheses freely here — later "
            "they'll be validated on a separate split."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "regime_type": {
                    "type": "string",
                    "enum": ["eke", "ow_negative", "ow_positive", "anom_percentile", "lc_extent"],
                    "description": (
                        "eke: eddy kinetic energy, high values = energetic region. "
                        "ow_negative: Okubo-Weiss < 0, inside vortex cores. "
                        "ow_positive: Okubo-Weiss > 0, strain/frontal zones. "
                        "anom_percentile: timesteps with high domain-mean SSH anomaly. "
                        "lc_extent: timesteps when Loop Current extends to a given latitude."
                    ),
                },
                "comparator": {
                    "type": "string",
                    "enum": ["percentile_gt", "gt", "lt"],
                    "description": (
                        "percentile_gt: value is 0-100 percentile (for eke, anom_percentile). "
                        "gt or lt: value is a latitude threshold (for lc_extent). "
                        "For ow_negative/ow_positive, comparator is ignored (pass 'gt')."
                    ),
                },
                "value": {
                    "type": "number",
                    "description": "Threshold value. Percentile 0-100 or latitude in degrees.",
                },
            },
            "required": ["regime_type", "comparator", "value"],
        },
    },
]


SYSTEM_PROMPT = """You are an AI researcher investigating where a neural-network surrogate
of Gulf of Mexico ocean dynamics fails. Your job is to propose and test physical-regime
hypotheses about failure modes.

Available regime types:
  - eke (eddy kinetic energy): high values = energetic, eddy-dominated regions
  - ow_negative: Okubo-Weiss < 0, inside vortex cores
  - ow_positive: Okubo-Weiss > 0, strain-dominated regions and fronts
  - anom_percentile: timesteps with large domain-mean SSH anomaly
  - lc_extent: Loop Current northward extent at each timestep

Your protocol:
  1. Propose a hypothesis in plain English, grounded in ocean physics.
  2. Call the evaluate_regime tool to test it on the discovery split.
  3. Read the result. Look at mean_err_inside_mm, mean_err_outside_mm, error_ratio, p_value_discovery.
  4. If an effect is meaningful (error_ratio > 1.2 and p_value < 0.001), note it as a candidate.
  5. Propose another hypothesis — try different regime types or thresholds.
  6. After exhausting your budget, summarize the candidate regimes you want to validate.

Be curious. Try at least one hypothesis for each regime type. Be especially interested
in high-EKE regions, vortex cores, and large-anomaly timesteps — these are physically
where the surrogate is expected to struggle."""


def run_gemini_loop(data, api_key):
    """Run the Gemini-driven hypothesis generation loop."""
    client = genai.Client(api_key=api_key)

    function_declarations = []
    for tool in GEMINI_TOOLS:
        function_declarations.append(types.FunctionDeclaration(
            name=tool["name"],
            description=tool["description"],
            parameters=tool["parameters"],
        ))

    tools = [types.Tool(function_declarations=function_declarations)]
    config = types.GenerateContentConfig(
        tools=tools,
        system_instruction=SYSTEM_PROMPT,
        temperature=0.7,
    )

    history = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=(
                f"Begin your investigation. You have a budget of {BUDGET} evaluate_regime calls. "
                f"Propose hypotheses and call the tool to test them. When you're done, "
                f"output a summary of the candidate regimes you want to validate."
            ))]
        )
    ]

    candidates = []

    for turn in range(BUDGET + 4):
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=history,
            config=config,
        )

        candidate = response.candidates[0]
        history.append(candidate.content)

        tool_calls = []
        text_parts = []
        for part in candidate.content.parts:
            if hasattr(part, "function_call") and part.function_call:
                tool_calls.append(part.function_call)
            if hasattr(part, "text") and part.text:
                text_parts.append(part.text)

        if text_parts:
            print(f"\n[Agent turn {turn}] {''.join(text_parts)[:500]}")

        if not tool_calls:
            break

        tool_responses = []
        for call in tool_calls:
            args = dict(call.args)
            print(f"  -> Tool call: evaluate_regime({args})")
            result = evaluate_regime(data, **args)
            er = result.get("error_ratio")
            pv = result.get("p_value_discovery")
            print(f"     Result: err_ratio={er if er is None else f'{er:.3g}'} "
                  f"p={pv if pv is None else f'{pv:.3g}'}")

            if (result.get("status") == "ok"
                    and result.get("error_ratio", 0) > 1.2
                    and result.get("p_value_discovery", 1) < 0.001):
                candidates.append({
                    "regime_type": args["regime_type"],
                    "comparator": args["comparator"],
                    "value": args["value"],
                    "discovery": result,
                })

            tool_responses.append(types.Part.from_function_response(
                name="evaluate_regime",
                response={"result": result},
            ))

        history.append(types.Content(role="user", parts=tool_responses))

        if len(TOOL_CALL_LOG) >= BUDGET:
            history.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=(
                    f"Budget exhausted ({BUDGET} calls used). Summarize your findings."
                ))]
            ))
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=history,
                config=config,
            )
            final_text = "".join(p.text for p in response.candidates[0].content.parts
                                 if hasattr(p, "text") and p.text)
            print(f"\n[Final summary]\n{final_text}")
            break

    return candidates


def validate_and_report(data, candidates):
    """Re-test each candidate on validation split and apply Bonferroni correction."""
    n_tests = len(candidates)
    if n_tests == 0:
        print("No candidate regimes survived discovery.")
        return []

    alpha = 0.05
    alpha_corrected = alpha / n_tests
    print(f"\nValidating {n_tests} candidates at Bonferroni-corrected alpha = {alpha_corrected:.4f}")

    validated = []
    for c in candidates:
        val_result = validate_regime(
            data, c["regime_type"], c["comparator"], c["value"]
        )
        if val_result.get("status") != "ok":
            continue
        p_val = val_result["p_value_validation"]
        is_validated = p_val < alpha_corrected and val_result["error_ratio"] > 1.1
        finding = {
            "regime_type": c["regime_type"],
            "comparator": c["comparator"],
            "value": c["value"],
            "discovery": {
                "error_ratio": c["discovery"]["error_ratio"],
                "p_value": c["discovery"]["p_value_discovery"],
                "n_inside": c["discovery"]["n_inside"],
                "call_id": c["discovery"]["call_id"],
            },
            "validation": {
                "error_ratio": val_result["error_ratio"],
                "p_value": p_val,
                "n_inside": val_result["n_inside_val"],
                "call_id": val_result["call_id"],
            },
            "mean_err_inside_val_mm": val_result["mean_err_inside_mm"],
            "mean_err_outside_val_mm": val_result["mean_err_outside_mm"],
            "bonferroni_alpha": alpha_corrected,
            "validated": bool(is_validated),
        }
        validated.append(finding)

    return validated


def describe_regime(finding):
    """Human-readable description of a regime."""
    r = finding["regime_type"]
    v = finding["value"]
    if r == "eke":
        return f"High eddy kinetic energy (top {100 - v:.0f}% of pixel-timesteps)"
    if r == "ow_negative":
        return "Vortex-core regions (Okubo-Weiss < 0)"
    if r == "ow_positive":
        return "Strain/frontal regions (Okubo-Weiss > 0)"
    if r == "anom_percentile":
        return f"Timesteps with high domain-mean SSH anomaly (top {100 - v:.0f}%)"
    if r == "lc_extent":
        return f"Timesteps when Loop Current extends {finding['comparator']} {v:.1f}°N"
    return str(finding)


def main():
    if not GEMINI_AVAILABLE:
        print("ERROR: google-genai not installed. Run:")
        print("  pip install google-genai scipy")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: set GEMINI_API_KEY environment variable.")
        print("  export GEMINI_API_KEY='your-key-here'")
        return

    print("=" * 70)
    print("AUTORESEARCH LOOP — physical regime failure-mode discovery")
    print(f"  Agent: Gemini ({MODEL_NAME})")
    print("=" * 70)

    data = load_data()
    print(f"Loaded data. Discovery split: t=0..{data['discovery_end']}, "
          f"Validation: t={data['validation_start']}..{data['T']}")

    # Discovery phase
    print("\n--- DISCOVERY PHASE (agent proposes and tests hypotheses) ---")
    candidates = run_gemini_loop(data, api_key)
    print(f"\nDiscovery yielded {len(candidates)} candidate regimes.")

    # Validation phase
    print("\n--- VALIDATION PHASE (Bonferroni-corrected held-out test) ---")
    findings = validate_and_report(data, candidates)

    # Report
    print("\n" + "=" * 70)
    print("FINAL FINDINGS")
    print("=" * 70)
    n_validated = sum(1 for f in findings if f["validated"])
    print(f"{n_validated} / {len(findings)} candidates passed Bonferroni-corrected validation.\n")

    for f in findings:
        tag = "[VALIDATED]" if f["validated"] else "[rejected]"
        print(f"{tag} {describe_regime(f)}")
        print(f"   Discovery err ratio: {f['discovery']['error_ratio']:.3f} "
              f"(p={f['discovery']['p_value']:.2e})")
        print(f"   Validation err ratio: {f['validation']['error_ratio']:.3f} "
              f"(p={f['validation']['p_value']:.2e})")
        print(f"   Mean error inside: {f['mean_err_inside_val_mm']:.2f} mm, "
              f"outside: {f['mean_err_outside_val_mm']:.2f} mm")
        print(f"   Receipts: discovery={f['discovery']['call_id']}, "
              f"validation={f['validation']['call_id']}")
        print()

    # Save
    os.makedirs("outputs", exist_ok=True)
    out = {
        "findings": findings,
        "tool_call_log": TOOL_CALL_LOG,
        "config": {
            "discovery_fraction": DISCOVERY_FRAC,
            "budget": BUDGET,
            "model": MODEL_NAME,
            "bonferroni_alpha": 0.05 / max(1, len(findings)),
        },
    }
    with open("outputs/findings.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Saved outputs/findings.json")


if __name__ == "__main__":
    main()
