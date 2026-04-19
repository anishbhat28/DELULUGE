import re
import subprocess
import sys
import time
from pathlib import Path

import streamlit as st

from automated_preprocessing import extract_train_context, extract_data_features, build_prompt
from openai import OpenAI as _OpenAI

AUTORESEARCH_BUDGET = 10

ALLOWED_DATA_EXTENSIONS  = {"csv"}
ALLOWED_MODEL_EXTENSIONS = {"py"}

st.set_page_config(page_title="Model Diagnostics", page_icon="📊", layout="centered")

if "last_run_sig" not in st.session_state:
    st.session_state.last_run_sig = None

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: #faf6ee !important;
    min-height: 100vh;
    position: relative !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 760px !important; }

/* ── Nav ── */
.nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 18px 0;
    border-bottom: 1px solid #e8dff5;
}
.nav-left { display: flex; align-items: center; gap: 10px; }
.nav-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #a894e0, #cfc0f5);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.nav-title { color: #3b3040; font-weight: 600; font-size: 1.05rem; letter-spacing: -.2px; }
.nav-badge {
    background: #f0ebfc;
    color: #8b6fd4;
    border: 1px solid #d8ccf4;
    font-size: .7rem; font-weight: 600;
    padding: 2px 10px; border-radius: 20px;
    text-transform: uppercase; letter-spacing: .5px;
}

/* ── Hero ── */
.hero { text-align: center; padding: 48px 20px 36px; }
.hero-eyebrow {
    display: inline-block;
    background: #f0ebfc;
    border: 1px solid #d8ccf4;
    color: #8b6fd4;
    font-size: .75rem; font-weight: 600;
    padding: 5px 14px; border-radius: 20px;
    margin-bottom: 18px; letter-spacing: .5px; text-transform: uppercase;
}
.hero h1 { font-size: 2.4rem; font-weight: 700; line-height: 1.15; letter-spacing: -.5px; margin-bottom: 12px; color: #3b3040; }
.hero h1 span { color: #8b6fd4; }
.hero p { font-size: 1rem; color: #9a8fa6; line-height: 1.65; }

/* ── Card ── */
.card {
    background: #fffdf9;
    border-radius: 20px;
    border: 1px solid #ecdff5;
    box-shadow: 0 4px 24px rgba(180,160,210,.12);
    overflow: hidden; margin-bottom: 32px;
}
.card-header {
    background: linear-gradient(90deg, #f5effc, #fffdf9);
    border-bottom: 1px solid #ecdff5;
    padding: 22px 30px;
    display: flex; align-items: center; gap: 12px;
}
.card-header-icon {
    width: 40px; height: 40px;
    background: #ece4fa; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.card-header h2 { font-size: 1.05rem; font-weight: 600; color: #3b3040; margin: 0; }
.card-header p  { font-size: .82rem; color: #8b6fd4; margin: 2px 0 0; }
.card-body { padding: 28px 30px 30px; }

/* ── Steps ── */
.steps { display: flex; gap: 0; margin-bottom: 28px; position: relative; }
.steps::before {
    content: '';
    position: absolute; top: 14px; left: 14%; right: 14%;
    height: 2px; background: #e8dff5; z-index: 0;
}
.step { flex: 1; display: flex; flex-direction: column; align-items: center; gap: 6px; position: relative; z-index: 1; }
.step-num {
    width: 28px; height: 28px; border-radius: 50%;
    background: #a894e0; color: #fff;
    font-size: .75rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.step-label { font-size: .72rem; color: #8b6fd4; font-weight: 600; }

/* ── Labels ── */
.field-label { font-size: .83rem; font-weight: 600; color: #4e4260; margin-bottom: 2px; }
.field-hint  { font-size: .76rem; color: #9a8fa6; margin-bottom: 6px; }

/* ── Streamlit widget overrides ── */
[data-testid="stFileUploader"] {
    border: 2px solid #d0c2f0 !important;
    border-radius: 12px !important;
    background: #faf5ff !important;
    transition: border-color .2s, background .2s !important;
    overflow: hidden !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #a894e0 !important;
    background: #f3ecfc !important;
}
[data-testid="stFileUploader"] label { display: none !important; }

[data-testid="stFileUploaderDropzone"] {
    background: #faf5ff !important;
    border: none !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"]:hover { background: #f3ecfc !important; }

[data-testid="stFileUploaderDropzoneInstructions"] { color: #8b6fd4 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small,
[data-testid="stFileUploaderDropzoneInstructions"] * { color: #8b6fd4 !important; }
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] div,
[data-testid="stFileUploaderDropzone"] small { color: #8b6fd4 !important; }
[data-testid="stFileUploaderDropzoneInstructions"] svg { fill: #a894e0 !important; color: #a894e0 !important; }
[data-testid="stFileUploader"] button {
    background: #ede5fc !important; color: #4e3a80 !important;
    border: 1px solid #d0c2f0 !important; border-radius: 6px !important; font-weight: 600 !important;
}
[data-testid="stFileUploader"] button:hover { background: #e0d4fa !important; }
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] div,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploaderFile"] span,
[data-testid="stFileUploaderFile"] *,
[data-testid="stUploadedFile"] span,
[data-testid="stUploadedFile"] * { color: #8b6fd4 !important; }

[data-testid="stTextArea"] textarea {
    border: 1.5px solid #cfc0f5 !important;
    border-radius: 8px !important;
    background: #fdf9ff !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .9rem !important; color: #3b3040 !important;
    caret-color: #000 !important;
    transition: border-color .15s, box-shadow .15s !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #a894e0 !important;
    box-shadow: 0 0 0 3px rgba(196,181,244,.18) !important;
}
[data-testid="stTextArea"] label { display: none !important; }

/* ── Button ── */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #a894e0, #cfc0f5) !important;
    color: #4e3a80 !important; border: none !important;
    border-radius: 8px !important; padding: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .95rem !important; font-weight: 600 !important;
    letter-spacing: -.1px !important;
    box-shadow: 0 2px 10px rgba(196,181,244,.35) !important;
    transition: all .2s !important; margin-top: 8px !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 18px rgba(196,181,244,.5) !important;
    background: linear-gradient(135deg, #b8a6f0, #d4c8f8) !important;
}

/* ── Response panels ── */
.resp-success { border: 1.5px solid #a8d5b8; border-radius: 8px; overflow: hidden; margin-top: 20px; }
.resp-success-header { background: #d8f2e4; color: #3a8a5c; padding: 10px 16px; font-size: .83rem; font-weight: 600; }
.resp-error { border: 1.5px solid #f4afc0; border-radius: 8px; overflow: hidden; margin-top: 20px; }
.resp-error-header { background: #fce8ef; color: #c05070; padding: 10px 16px; font-size: .83rem; font-weight: 600; }
.resp-body { background: #fffdf9; padding: 16px; font-size: .85rem; line-height: 1.7; color: #4e4260; }
.resp-row { display: flex; gap: 10px; margin-bottom: 5px; }
.resp-key { font-weight: 600; color: #a894e0; min-width: 80px; font-size: .78rem; text-transform: uppercase; letter-spacing: .3px; }
.resp-val { color: #3b3040; }

.site-footer { text-align: center; padding: 16px 0 24px; font-size: .75rem; color: #c4b8d0; }

</style>
""", unsafe_allow_html=True)

# ── Nav ──
st.markdown("""
<div class="nav">
  <div class="nav-left">
    <div class="nav-icon">📊</div>
    <span class="nav-title">Model Diagnostics</span>
  </div>
  <span class="nav-badge">Research Tool</span>
</div>
""", unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
  <div class="hero-eyebrow">AI-Powered Analysis</div>
  <h1>Interpret Your <span>Model Output</span></h1>
  <p>Upload a trained model and a dataset, then describe what you want to know.</p>
</div>
""", unsafe_allow_html=True)

# ── Card ──
st.markdown("""
<div class="card">
  <div class="card-header">
    <div class="card-header-icon">🗂️</div>
    <div>
      <h2>New Session</h2>
      <p>Complete all three steps to run your analysis</p>
    </div>
  </div>
  <div class="card-body">
    <div class="steps">
      <div class="step"><div class="step-num">1</div><div class="step-label">Training Script</div></div>
      <div class="step"><div class="step-num">2</div><div class="step-label">Upload Data</div></div>
      <div class="step"><div class="step-num">3</div><div class="step-label">Write Prompt</div></div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="field-label">Training Script <span style="color:#e08898">*</span></div>', unsafe_allow_html=True)
st.markdown('<div class="field-hint">Your train.py will be executed with <code>python train.py</code>. It must read <code>data.csv</code> and write <code>predictions.csv</code> with columns <code>target</code> + <code>prediction</code> (plus optional numeric feature columns).</div>', unsafe_allow_html=True)
modelfile = st.file_uploader("Training script", type=list(ALLOWED_MODEL_EXTENSIONS), key="model")

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

st.markdown('<div class="field-label">Data File <span style="color:#e08898">*</span></div>', unsafe_allow_html=True)
st.markdown('<div class="field-hint">Tabular CSV. Your train.py consumes this and produces predictions.csv.</div>', unsafe_allow_html=True)
datafile = st.file_uploader("Data file", type=list(ALLOWED_DATA_EXTENSIONS), key="data")

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

st.markdown('<div class="field-label">Prompt <span style="color:#e08898">*</span></div>', unsafe_allow_html=True)
st.markdown('<div class="field-hint">Describe what you want GPT to analyze or explain about the model output.</div>', unsafe_allow_html=True)
prompt = st.text_area(
    "Prompt",
    placeholder="e.g. What are the most significant predictors in this model, and how do they interact with each other?",
    height=130,
    max_chars=2000,
)

st.markdown('</div></div>', unsafe_allow_html=True)

current_sig = (
    modelfile.name if modelfile else None,
    datafile.name if datafile else None,
    prompt.strip(),
)
already_run = current_sig == st.session_state.last_run_sig
run = st.button("Run Analysis →", disabled=already_run)

if run:
    errors = []
    if not modelfile:
        errors.append("No model file provided.")
    elif not ("." in modelfile.name and modelfile.name.rsplit(".", 1)[1].lower() in ALLOWED_MODEL_EXTENSIONS):
        errors.append(f"Model file type not supported. Accepted: {', '.join(ALLOWED_MODEL_EXTENSIONS)}")
    if not datafile:
        errors.append("No data file provided.")
    elif not ("." in datafile.name and datafile.name.rsplit(".", 1)[1].lower() in ALLOWED_DATA_EXTENSIONS):
        errors.append(f"Data file type not supported. Accepted: {', '.join(ALLOWED_DATA_EXTENSIONS)}")
    if not prompt.strip():
        errors.append("Prompt cannot be empty.")

    if errors:
        for e in errors:
            st.markdown(f"""
            <div class="resp-error">
              <div class="resp-error-header">Error</div>
              <div class="resp-body">{e}</div>
            </div>""", unsafe_allow_html=True)
    else:
        project_root = Path(__file__).parent

        def save_upload(f, dest: Path):
            with open(dest, "wb") as out:
                out.write(f.getbuffer())
            return dest

        train_path = save_upload(modelfile, project_root / "train.py")
        data_ext = datafile.name.rsplit(".", 1)[1].lower()
        data_path = save_upload(datafile, project_root / f"data.{data_ext}")
        output_path = project_root / "program.md"

        # Contract shims — make the uploaded train.py produce predictions.csv
        # and read whatever filename it hardcoded by aliasing common names.
        import shutil as _shutil
        for alias in ("combined_data.csv", "input_data.csv", "train_data.csv"):
            _alias_path = project_root / alias
            if not _alias_path.exists():
                _shutil.copy(data_path, _alias_path)

        _train_src = train_path.read_text(encoding="utf-8")
        if "predictions.csv" not in _train_src:
            _train_src += """

# --- auto-injected predictions.csv writer (contract shim) ---
try:
    import pandas as _pd
    from pathlib import Path as _Path
    _g = globals()
    _src_df = _g.get('df') if isinstance(_g.get('df'), _pd.DataFrame) else None
    _feats = _g.get('features')
    _preds_all = None
    for _k in ('rf_preds_all', 'lr_preds_all', 'preds_all', 'predictions', 'y_pred'):
        _v = _g.get(_k)
        if _v is not None:
            _preds_all = _v
            break
    _y_all = _g.get('y_all')
    if _y_all is None and _src_df is not None:
        for _k in ('sea_level_mm', 'target', 'y'):
            if _k in _src_df.columns:
                _y_all = _src_df[_k]
                break
    if _src_df is not None and _feats is not None and _preds_all is not None and _y_all is not None:
        _keep = [c for c in (['year'] + list(_feats)) if c in _src_df.columns]
        _out = _src_df[_keep].copy()
        _out['target'] = _y_all.values if hasattr(_y_all, 'values') else _y_all
        _out['prediction'] = _preds_all
        _out.to_csv(_Path(__file__).parent / 'predictions.csv', index=False)
        print(f"auto-injected: wrote predictions.csv ({len(_out)} rows)")
    else:
        print("auto-inject skipped: could not locate df/features/preds/y_all in globals")
except Exception as _e:
    print(f"auto-inject skipped: {_e}")
"""
            train_path.write_text(_train_src, encoding="utf-8")

        try:
            bar = st.progress(0, text="Saving uploaded files…")
            bar.progress(8, text="Parsing training script…")
            train_context = extract_train_context(train_path)
            bar.progress(15, text="Extracting data features…")
            data_features = extract_data_features(data_path)
            bar.progress(22, text="Building prompt…")
            prompt_text = build_prompt(train_context, prompt.strip(), data_features)
            bar.progress(30, text="Calling GPT API — this may take a moment…")
            _client = _OpenAI()
            response = _client.responses.create(model="gpt-5.4", input=prompt_text)
            program_md = response.output_text.strip()
            if program_md.startswith("```"):
                lines = program_md.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                program_md = "\n".join(lines).strip()
            bar.progress(35, text="Writing program.md…")
            output_path.write_text(program_md, encoding="utf-8")
            bar.progress(38, text="program.md saved. Running your training script…")

            st.markdown("""
            <div class="resp-success">
              <div class="resp-success-header">✓ program.md saved — executing train.py</div>
            </div>""", unsafe_allow_html=True)

            log_box = st.empty()
            log_lines: list[str] = []

            def push_log(line: str):
                log_lines.append(line)
                if len(log_lines) > 25:
                    del log_lines[: len(log_lines) - 25]
                log_box.code("\n".join(log_lines), language=None)

            predictions_path = project_root / "predictions.csv"
            if predictions_path.exists():
                predictions_path.unlink()

            train_proc = subprocess.Popen(
                [sys.executable, "train.py"],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert train_proc.stdout is not None
            for raw in train_proc.stdout:
                push_log(raw.rstrip())
            train_proc.wait()

            if train_proc.returncode != 0:
                st.markdown(f"""
                <div class="resp-error">
                  <div class="resp-error-header">train.py failed (exit {train_proc.returncode})</div>
                  <div class="resp-body">See log above. Contract: train.py must read data.csv and write predictions.csv with columns target + prediction.</div>
                </div>""", unsafe_allow_html=True)
                st.stop()

            if not predictions_path.exists():
                st.markdown("""
                <div class="resp-error">
                  <div class="resp-error-header">train.py ran but produced no predictions.csv</div>
                  <div class="resp-body">Contract: train.py must write <code>predictions.csv</code> in the project root with columns <code>target</code> and <code>prediction</code> (plus optional numeric feature columns).</div>
                </div>""", unsafe_allow_html=True)
                st.stop()

            bar.progress(50, text="Training complete. Launching deep research…")

            proc = subprocess.Popen(
                [
                    sys.executable, "autoresearch.py",
                    "--data", str(predictions_path),
                    "--prompt", prompt.strip(),
                ],
                cwd=str(project_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            turn_re = re.compile(r"\[Agent turn (\d+)\]")
            assert proc.stdout is not None
            for raw in proc.stdout:
                line = raw.rstrip()
                push_log(line)
                if "Loaded data" in line:
                    bar.progress(55, text="Deep research: data loaded…")
                elif "DISCOVERY PHASE" in line:
                    bar.progress(58, text="Deep research: discovery phase…")
                elif (m := turn_re.search(line)):
                    turn = int(m.group(1))
                    pct = min(58 + int((turn / AUTORESEARCH_BUDGET) * 30), 88)
                    bar.progress(pct, text=f"Deep research: hypothesis {turn + 1}/{AUTORESEARCH_BUDGET}…")
                elif "VALIDATION PHASE" in line:
                    bar.progress(90, text="Deep research: validating on held-out split…")
                elif "FINAL FINDINGS" in line:
                    bar.progress(95, text="Deep research: compiling findings…")
                elif "Saved outputs/findings.json" in line:
                    bar.progress(98, text="Deep research: findings saved.")

            proc.wait()

            if proc.returncode != 0:
                st.markdown(f"""
                <div class="resp-error">
                  <div class="resp-error-header">Autoresearch failed (exit {proc.returncode})</div>
                  <div class="resp-body">See log above.</div>
                </div>""", unsafe_allow_html=True)
            else:
                bar.progress(100, text="Complete! Opening dashboard…")
                st.session_state.last_run_sig = current_sig
                time.sleep(0.6)
                st.switch_page("dashboard.py")
        except Exception as e:
            st.markdown(f"""
            <div class="resp-error">
              <div class="resp-error-header">Error</div>
              <div class="resp-body">{e}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="site-footer">Model Diagnostics &middot; Research Tool &middot; 2026</div>', unsafe_allow_html=True)
