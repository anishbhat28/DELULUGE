"""
autoresearch.py
===============

Grounded autoresearch loop for generic tabular failure-mode discovery.

Flow:
    1. Gemini proposes a regime hypothesis in natural language.
    2. Gemini calls a tool to test it on a DISCOVERY split.
    3. Accepted hypotheses are re-tested on a held-out VALIDATION split
       the agent never saw during generation.
    4. Bonferroni correction applied to the total hypothesis count.
    5. Output: outputs/findings.json.

Data source: a tabular CSV of predictions + targets (+ optional features),
loaded via rmse_regimes.load_tabular. Regime fields are provided by
rmse_regimes.compute_regime_fields — see that file for the contract.

Usage:
    python autoresearch.py [--data path/to/data.csv]
"""

import argparse
import json
import os
import uuid

import numpy as np
from scipy import stats

from rmse_regimes import compute_regime_fields, load_tabular

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


DISCOVERY_FRAC = 0.7
BUDGET = 10
MODEL_NAME = "gemini-2.5-flash"

TOOL_CALL_LOG: list[dict] = []


def load_data(data_path: str) -> dict:
    bundle = load_tabular(data_path)
    regimes = compute_regime_fields(bundle)
    n = int(len(bundle["target"]))
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    cutoff = int(n * DISCOVERY_FRAC)
    return {
        "abs_error": bundle["abs_error"],
        "regimes": regimes,
        "disc_idx": perm[:cutoff],
        "val_idx": perm[cutoff:],
        "n": n,
        "regime_field_names": list(regimes.keys()),
        "feature_names": list(bundle["features"].columns),
        "target_col": bundle["target_col"],
        "pred_col": bundle["pred_col"],
    }


def build_mask(data: dict, regime_field: str, comparator: str, value: float, split: str):
    if regime_field not in data["regimes"]:
        raise ValueError(
            f"unknown regime_field '{regime_field}'. Available: {data['regime_field_names']}"
        )
    idx = data["disc_idx"] if split == "discovery" else data["val_idx"]
    field = data["regimes"][regime_field][idx]

    if comparator == "percentile_gt":
        thresh = float(np.percentile(field, value))
        mask = field > thresh
        meta = {"threshold": thresh, "percentile": value}
    elif comparator == "percentile_lt":
        thresh = float(np.percentile(field, value))
        mask = field < thresh
        meta = {"threshold": thresh, "percentile": value}
    elif comparator == "gt":
        mask = field > value
        meta = {"threshold": float(value)}
    elif comparator == "lt":
        mask = field < value
        meta = {"threshold": float(value)}
    elif comparator == "eq":
        mask = field == value
        meta = {"threshold": float(value)}
    else:
        raise ValueError(f"unknown comparator '{comparator}'")
    return mask, meta, idx


def _regime_test(data: dict, regime_field: str, comparator: str, value: float, split: str, call_type: str) -> dict:
    call_id = str(uuid.uuid4())[:8]
    try:
        mask, meta, idx = build_mask(data, regime_field, comparator, value, split)
        err = data["abs_error"][idx]
        inside = err[mask]
        outside = err[~mask]
        if inside.size < 5 or outside.size < 5:
            result = {
                "call_id": call_id,
                "status": "insufficient_data",
                "n_inside": int(inside.size),
                "n_outside": int(outside.size),
            }
        else:
            t_stat, p_val = stats.ttest_ind(inside, outside, equal_var=False)
            result = {
                "call_id": call_id,
                "status": "ok",
                "regime_field": regime_field,
                "comparator": comparator,
                "value": value,
                "meta": meta,
                "n_inside": int(inside.size),
                "n_outside": int(outside.size),
                "mean_err_inside": float(inside.mean()),
                "mean_err_outside": float(outside.mean()),
                "error_ratio": float(inside.mean() / (outside.mean() + 1e-12)),
                "t_statistic": float(t_stat),
                "p_value": float(p_val),
            }
        TOOL_CALL_LOG.append({
            "call_id": call_id,
            "type": call_type,
            "input": {"regime_field": regime_field, "comparator": comparator, "value": value},
            "output": result,
        })
        return result
    except Exception as e:
        result = {"call_id": call_id, "status": "error", "error": str(e)}
        TOOL_CALL_LOG.append({"call_id": call_id, "type": call_type, "error": str(e)})
        return result


def evaluate_regime(data, regime_field, comparator, value):
    """TOOL. Evaluate on discovery split."""
    r = _regime_test(data, regime_field, comparator, value, "discovery", "evaluate_regime")
    # surface the key the agent reads as p_value_discovery for clarity
    if r.get("status") == "ok":
        r["p_value_discovery"] = r["p_value"]
    return r


def validate_regime(data, regime_field, comparator, value):
    r = _regime_test(data, regime_field, comparator, value, "validation", "validate_regime")
    if r.get("status") == "ok":
        r["p_value_validation"] = r["p_value"]
    return r


def build_gemini_tools(data: dict):
    """Build the Gemini function declaration, listing the regime fields this dataset exposes."""
    field_list = "\n".join(f"  - {name}" for name in data["regime_field_names"])
    description = (
        "Test whether the model's absolute error is systematically higher inside "
        "a specified regime than outside it. Uses a held-out DISCOVERY split of "
        "the data so you can test multiple hypotheses here — validation happens "
        "later on a separate split.\n\n"
        "Available regime fields for this dataset:\n" + field_list
    )
    parameters = {
        "type": "object",
        "properties": {
            "regime_field": {
                "type": "string",
                "enum": data["regime_field_names"],
                "description": (
                    "Regime field to slice on. 'target'/'prediction'/'abs_error'/"
                    "'residual'/'residual_sign' are always available. 'feature::<col>' "
                    "entries come from the uploaded CSV's numeric feature columns."
                ),
            },
            "comparator": {
                "type": "string",
                "enum": ["percentile_gt", "percentile_lt", "gt", "lt", "eq"],
                "description": (
                    "percentile_gt / percentile_lt: value is 0-100 percentile. "
                    "gt / lt / eq: value is a raw threshold."
                ),
            },
            "value": {
                "type": "number",
                "description": "Percentile 0-100 or raw threshold.",
            },
        },
        "required": ["regime_field", "comparator", "value"],
    }
    return [{"name": "evaluate_regime", "description": description, "parameters": parameters}]


SYSTEM_PROMPT_TEMPLATE = """You are an AI researcher investigating where a machine-learning model
systematically makes larger absolute errors. You have access to a tabular dataset with
target column '{target_col}', prediction column '{pred_col}', and these numeric feature
columns: {feature_cols}.
{user_focus}
Protocol:
  1. Propose a hypothesis in plain English, grounded in the data.
  2. Call the evaluate_regime tool to test it on the discovery split.
  3. Look at mean_err_inside, mean_err_outside, error_ratio, p_value_discovery.
  4. If error_ratio > 1.2 and p_value_discovery < 0.001, note it as a candidate.
  5. Propose another hypothesis — try different regime fields / comparators / thresholds.
  6. When your budget is spent, summarize the candidate regimes you want validated.

Be systematic: test high-error vs low-error regions, extreme-target percentiles,
extreme-prediction percentiles, residual_sign (over- vs under-prediction bias),
and each feature::<col> where a monotonic error trend is plausible."""


def run_gemini_loop(data, api_key, user_prompt: str = ""):
    client = genai.Client(api_key=api_key)

    tool_specs = build_gemini_tools(data)
    function_declarations = [
        types.FunctionDeclaration(name=t["name"], description=t["description"], parameters=t["parameters"])
        for t in tool_specs
    ]
    tools = [types.Tool(function_declarations=function_declarations)]

    focus_block = (
        f"\nThe user has specifically asked:\n  {user_prompt.strip()}\n"
        "Let that question steer which regimes you prioritize, but still test broadly.\n"
        if user_prompt.strip() else ""
    )
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        target_col=data["target_col"],
        pred_col=data["pred_col"],
        feature_cols=(", ".join(data["feature_names"]) or "(none)"),
        user_focus=focus_block,
    )
    config = types.GenerateContentConfig(
        tools=tools, system_instruction=system_prompt, temperature=0.7,
    )

    history = [types.Content(
        role="user",
        parts=[types.Part.from_text(text=(
            f"Begin your investigation. You have a budget of {BUDGET} evaluate_regime calls. "
            f"Propose hypotheses and call the tool to test them. When you're done, "
            f"output a summary of the candidate regimes you want to validate."
        ))]
    )]
    candidates = []

    for turn in range(BUDGET + 4):
        response = client.models.generate_content(model=MODEL_NAME, contents=history, config=config)
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
                    "regime_field": args["regime_field"],
                    "comparator": args["comparator"],
                    "value": args["value"],
                    "discovery": result,
                })

            tool_responses.append(types.Part.from_function_response(
                name="evaluate_regime",
                response={"result": result},
            ))

        history.append(types.Content(role="user", parts=tool_responses))

        if len([c for c in TOOL_CALL_LOG if c.get("type") == "evaluate_regime"]) >= BUDGET:
            history.append(types.Content(
                role="user",
                parts=[types.Part.from_text(text=(
                    f"Budget exhausted ({BUDGET} calls used). Summarize your findings."
                ))]
            ))
            response = client.models.generate_content(model=MODEL_NAME, contents=history, config=config)
            final_text = "".join(
                p.text for p in response.candidates[0].content.parts
                if hasattr(p, "text") and p.text
            )
            print(f"\n[Final summary]\n{final_text}")
            break

    return candidates


def validate_and_report(data, candidates):
    n_tests = len(candidates)
    if n_tests == 0:
        print("No candidate regimes survived discovery.")
        return []

    alpha = 0.05
    alpha_corrected = alpha / n_tests
    print(f"\nValidating {n_tests} candidates at Bonferroni-corrected alpha = {alpha_corrected:.4f}")

    validated = []
    for c in candidates:
        val_result = validate_regime(data, c["regime_field"], c["comparator"], c["value"])
        if val_result.get("status") != "ok":
            continue
        p_val = val_result["p_value_validation"]
        is_validated = p_val < alpha_corrected and val_result["error_ratio"] > 1.1
        validated.append({
            "regime_field": c["regime_field"],
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
                "n_inside": val_result["n_inside"],
                "call_id": val_result["call_id"],
            },
            "mean_err_inside_val": val_result["mean_err_inside"],
            "mean_err_outside_val": val_result["mean_err_outside"],
            "bonferroni_alpha": alpha_corrected,
            "validated": bool(is_validated),
        })
    return validated


def describe_regime(finding) -> str:
    r = finding["regime_field"]
    c = finding["comparator"]
    v = finding["value"]
    if c == "percentile_gt":
        return f"{r} > {v:.1f} percentile"
    if c == "percentile_lt":
        return f"{r} < {v:.1f} percentile"
    return f"{r} {c} {v}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.csv", help="Path to tabular CSV")
    parser.add_argument("--prompt", default="", help="Optional user prompt to focus the agent")
    args = parser.parse_args()

    if not GEMINI_AVAILABLE:
        print("ERROR: google-genai not installed. Run:")
        print("  pip install google-genai scipy pandas")
        return

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: set GEMINI_API_KEY environment variable.")
        print("  export GEMINI_API_KEY='your-key-here'")
        return

    print("=" * 70)
    print("AUTORESEARCH LOOP — tabular failure-mode discovery")
    print(f"  Agent: Gemini ({MODEL_NAME})")
    print(f"  Data:  {args.data}")
    print("=" * 70)

    data = load_data(args.data)
    print(f"Loaded data: {data['n']} rows, {len(data['disc_idx'])} discovery / {len(data['val_idx'])} validation")
    print(f"Regime fields exposed: {data['regime_field_names']}")

    print("\n--- DISCOVERY PHASE (agent proposes and tests hypotheses) ---")
    candidates = run_gemini_loop(data, api_key, user_prompt=args.prompt)
    print(f"\nDiscovery yielded {len(candidates)} candidate regimes.")

    print("\n--- VALIDATION PHASE (Bonferroni-corrected held-out test) ---")
    findings = validate_and_report(data, candidates)

    print("\n" + "=" * 70)
    print("FINAL FINDINGS")
    print("=" * 70)
    n_validated = sum(1 for f in findings if f["validated"])
    print(f"{n_validated} / {len(findings)} candidates passed Bonferroni-corrected validation.\n")

    for f in findings:
        tag = "[VALIDATED]" if f["validated"] else "[rejected]"
        print(f"{tag} {describe_regime(f)}")
        print(f"   Discovery err ratio: {f['discovery']['error_ratio']:.3f} (p={f['discovery']['p_value']:.2e})")
        print(f"   Validation err ratio: {f['validation']['error_ratio']:.3f} (p={f['validation']['p_value']:.2e})")
        print(f"   Mean error inside: {f['mean_err_inside_val']:.4g}, outside: {f['mean_err_outside_val']:.4g}")
        print(f"   Receipts: discovery={f['discovery']['call_id']}, validation={f['validation']['call_id']}")
        print()

    os.makedirs("outputs", exist_ok=True)
    out = {
        "findings": findings,
        "tool_call_log": TOOL_CALL_LOG,
        "config": {
            "data_path": args.data,
            "target_col": data["target_col"],
            "pred_col": data["pred_col"],
            "feature_cols": data["feature_names"],
            "discovery_fraction": DISCOVERY_FRAC,
            "budget": BUDGET,
            "model": MODEL_NAME,
            "bonferroni_alpha": 0.05 / max(1, len(findings)),
        },
    }
    with open("outputs/findings.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("Saved outputs/findings.json")


if __name__ == "__main__":
    main()
