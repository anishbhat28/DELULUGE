import os
import streamlit as st
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
ALLOWED_DATA_EXTENSIONS  = {"csv", "json", "xlsx", "txt", "parquet"}
ALLOWED_MODEL_EXTENSIONS = {"pkl", "joblib", "h5", "pt", "pth", "onnx", "bin", "model"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="Model Diagnostics", page_icon="📊", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f1e3d 0%, #0f172a 60%, #0d1b2a 100%) !important;
    min-height: 100vh;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 760px !important; }

/* ── Nav bar ── */
.nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 0 18px;
    border-bottom: 1px solid rgba(255,255,255,.08);
    margin-bottom: 0;
}
.nav-left { display: flex; align-items: center; gap: 10px; }
.nav-icon {
    width: 36px; height: 36px;
    background: linear-gradient(135deg, #3b82f6, #60a5fa);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.nav-title { color: #fff; font-weight: 600; font-size: 1.05rem; letter-spacing: -.2px; }
.nav-badge {
    background: rgba(59,130,246,.25);
    color: #60a5fa;
    border: 1px solid rgba(59,130,246,.3);
    font-size: .7rem; font-weight: 600;
    padding: 2px 10px; border-radius: 20px;
    text-transform: uppercase; letter-spacing: .5px;
}

/* ── Hero ── */
.hero {
    text-align: center;
    padding: 48px 20px 36px;
    color: #fff;
}
.hero-eyebrow {
    display: inline-block;
    background: rgba(59,130,246,.15);
    border: 1px solid rgba(59,130,246,.25);
    color: #60a5fa;
    font-size: .75rem; font-weight: 600;
    padding: 5px 14px; border-radius: 20px;
    margin-bottom: 18px;
    letter-spacing: .5px; text-transform: uppercase;
}
.hero h1 {
    font-size: 2.4rem; font-weight: 700;
    line-height: 1.15; letter-spacing: -.5px;
    margin-bottom: 12px;
}
.hero h1 span { color: #60a5fa; }
.hero p { font-size: 1rem; color: rgba(255,255,255,.55); line-height: 1.65; }

/* ── Card ── */
.card {
    background: #fff;
    border-radius: 20px;
    box-shadow: 0 8px 32px rgba(0,0,0,.22);
    overflow: hidden;
    margin-bottom: 32px;
}
.card-header {
    background: linear-gradient(90deg, #eff6ff, #fff);
    border-bottom: 1px solid #f1f5f9;
    padding: 22px 30px;
    display: flex; align-items: center; gap: 12px;
}
.card-header-icon {
    width: 40px; height: 40px;
    background: #dbeafe; border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px;
}
.card-header h2 { font-size: 1.05rem; font-weight: 600; color: #0f172a; margin: 0; }
.card-header p  { font-size: .82rem; color: #64748b; margin: 2px 0 0; }
.card-body { padding: 28px 30px 30px; }

/* ── Steps ── */
.steps {
    display: flex; gap: 0;
    margin-bottom: 28px;
    position: relative;
}
.steps::before {
    content: '';
    position: absolute;
    top: 14px; left: 14%; right: 14%;
    height: 2px; background: #f1f5f9; z-index: 0;
}
.step {
    flex: 1; display: flex;
    flex-direction: column; align-items: center; gap: 6px;
    position: relative; z-index: 1;
}
.step-num {
    width: 28px; height: 28px; border-radius: 50%;
    background: #1d4ed8; color: #fff;
    font-size: .75rem; font-weight: 700;
    display: flex; align-items: center; justify-content: center;
}
.step-label { font-size: .72rem; color: #1d4ed8; font-weight: 600; }

/* ── Section labels ── */
.field-label {
    font-size: .83rem; font-weight: 600;
    color: #334155; margin-bottom: 2px;
}
.field-hint { font-size: .76rem; color: #64748b; margin-bottom: 6px; }

/* Streamlit widget overrides */
[data-testid="stFileUploader"] {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    background: #f8fafc !important;
    transition: border-color .2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
    background: #eff6ff !important;
}
[data-testid="stFileUploader"] label { display: none !important; }

[data-testid="stTextArea"] textarea {
    border: 1.5px solid #cbd5e1 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .9rem !important;
    color: #0f172a !important;
    transition: border-color .15s, box-shadow .15s !important;
}
[data-testid="stTextArea"] textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 3px rgba(59,130,246,.12) !important;
}
[data-testid="stTextArea"] label { display: none !important; }

/* Submit button */
[data-testid="stButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #1d4ed8, #3b82f6) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 14px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: .95rem !important; font-weight: 600 !important;
    letter-spacing: -.1px !important;
    box-shadow: 0 2px 8px rgba(29,78,216,.25) !important;
    transition: all .2s !important;
    margin-top: 8px !important;
}
[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(29,78,216,.35) !important;
}

/* Response panels */
.resp-success {
    border: 1.5px solid #86efac;
    border-radius: 8px; overflow: hidden;
    margin-top: 20px;
}
.resp-success-header {
    background: #dcfce7; color: #16a34a;
    padding: 10px 16px; font-size: .83rem; font-weight: 600;
}
.resp-error {
    border: 1.5px solid #fca5a5;
    border-radius: 8px; overflow: hidden;
    margin-top: 20px;
}
.resp-error-header {
    background: #fee2e2; color: #dc2626;
    padding: 10px 16px; font-size: .83rem; font-weight: 600;
}
.resp-body {
    background: #fff; padding: 16px;
    font-size: .85rem; line-height: 1.7; color: #334155;
}
.resp-row { display: flex; gap: 10px; margin-bottom: 5px; }
.resp-key {
    font-weight: 600; color: #94a3b8;
    min-width: 80px; font-size: .78rem;
    text-transform: uppercase; letter-spacing: .3px;
}
.resp-val { color: #1e293b; }

.site-footer {
    text-align: center; padding: 16px 0 24px;
    font-size: .75rem; color: rgba(255,255,255,.22);
}
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
  <p>Upload a trained model and a dataset, then describe what you want to know.<br>GPT will handle the interpretation.</p>
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
      <div class="step"><div class="step-num">1</div><div class="step-label">Upload Model</div></div>
      <div class="step"><div class="step-num">2</div><div class="step-label">Upload Data</div></div>
      <div class="step"><div class="step-num">3</div><div class="step-label">Write Prompt</div></div>
    </div>
""", unsafe_allow_html=True)

st.markdown('<div class="field-label">Model File <span style="color:#dc2626">*</span></div>', unsafe_allow_html=True)
st.markdown('<div class="field-hint">Accepted: pkl · joblib · h5 · pt · pth · onnx · bin · model</div>', unsafe_allow_html=True)
modelfile = st.file_uploader("Model file", type=list(ALLOWED_MODEL_EXTENSIONS), key="model")

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

st.markdown('<div class="field-label">Data File <span style="color:#dc2626">*</span></div>', unsafe_allow_html=True)
st.markdown('<div class="field-hint">Accepted: CSV · JSON · XLSX · TXT · Parquet — up to 1 GB</div>', unsafe_allow_html=True)
datafile = st.file_uploader("Data file", type=list(ALLOWED_DATA_EXTENSIONS), key="data")

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

st.markdown('<div class="field-label">Prompt <span style="color:#dc2626">*</span></div>', unsafe_allow_html=True)
st.markdown('<div class="field-hint">Describe what you want GPT to analyze or explain about the model output.</div>', unsafe_allow_html=True)
prompt = st.text_area(
    "Prompt",
    placeholder="e.g. What are the most significant predictors in this model, and how do they interact with each other?",
    height=130,
    max_chars=2000,
)

st.markdown('</div></div>', unsafe_allow_html=True)  # close card-body + card

run = st.button("Run Analysis →")

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
        def save_upload(f):
            fname = secure_filename(f.name)
            path = os.path.join(UPLOAD_FOLDER, fname)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            return fname

        model_filename = save_upload(modelfile)
        data_filename  = save_upload(datafile)

        # TODO: pass model_path, data_path, and prompt to your GPT pipeline here
        # result = run_gpt_pipeline(model_path=..., data_path=..., prompt=prompt)

        st.markdown(f"""
        <div class="resp-success">
          <div class="resp-success-header">✓ Request received</div>
          <div class="resp-body">
            <div class="resp-row"><span class="resp-key">Model</span><span class="resp-val">{model_filename}</span></div>
            <div class="resp-row"><span class="resp-key">Data</span><span class="resp-val">{data_filename}</span></div>
            <div class="resp-row"><span class="resp-key">Prompt</span><span class="resp-val">{prompt.strip()}</span></div>
            <div class="resp-row"><span class="resp-key">Status</span><span class="resp-val">Files received. GPT pipeline integration pending.</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="site-footer">Model Diagnostics &middot; Research Tool &middot; 2026</div>', unsafe_allow_html=True)
