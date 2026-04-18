import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
ALLOWED_DATA_EXTENSIONS  = {"csv", "json", "xlsx", "txt", "parquet"}
ALLOWED_MODEL_EXTENSIONS = {"pkl", "joblib", "h5", "pt", "pth", "onnx", "bin", "model"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1 GB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_data(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_DATA_EXTENSIONS


def allowed_model(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_MODEL_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyze", methods=["POST"])
def analyze():
    prompt     = request.form.get("prompt", "").strip()
    modelfile  = request.files.get("modelfile")
    datafile   = request.files.get("datafile")

    if not modelfile or modelfile.filename == "":
        return jsonify({"error": "No model file provided."}), 400
    if not allowed_model(modelfile.filename):
        return jsonify({"error": f"Model file type not supported. Accepted: {', '.join(ALLOWED_MODEL_EXTENSIONS)}"}), 400
    if not datafile or datafile.filename == "":
        return jsonify({"error": "No data file provided."}), 400
    if not allowed_data(datafile.filename):
        return jsonify({"error": f"Data file type not supported. Accepted: {', '.join(ALLOWED_DATA_EXTENSIONS)}"}), 400
    if not prompt:
        return jsonify({"error": "Prompt cannot be empty."}), 400

    model_filename = secure_filename(modelfile.filename)
    data_filename  = secure_filename(datafile.filename)

    model_path = os.path.join(app.config["UPLOAD_FOLDER"], model_filename)
    data_path  = os.path.join(app.config["UPLOAD_FOLDER"], data_filename)

    modelfile.save(model_path)
    datafile.save(data_path)

    # TODO: pass model_path, data_path, and prompt to your GPT pipeline here
    # result = run_gpt_pipeline(model_path=model_path, data_path=data_path, prompt=prompt)

    return jsonify({
        "status": "received",
        "model_file": model_filename,
        "data_file": data_filename,
        "prompt": prompt,
        "message": "Files received. GPT pipeline integration pending.",
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
