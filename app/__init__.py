"""Demo web app: upload CICIDS2017-format flow CSVs and get per-flow attack predictions (numpy inference)."""
import io
import os
from collections import Counter

import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from ids.infer import NumpyIDS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS = os.environ.get("IDS_ARTIFACTS", os.path.join(ROOT, "artifacts"))
SAMPLE = os.path.join(ROOT, "data", "sample_flows.csv")
MAX_ROWS = 50_000


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 40 * 1024 * 1024
    model = NumpyIDS(ARTIFACTS)

    def analyse(df):
        df = df.head(MAX_ROWS)
        df_clean, preds, conf = model.predict_frame(df)
        out = df_clean.rename(columns=lambda c: c.strip()).copy()
        out["Prediction"], out["Confidence"] = preds, conf.round(3)
        counts = Counter(preds)
        has_truth = "Label" in out.columns
        acc = None
        if has_truth:
            from ids.features import normalize_label
            acc = round(float((out["Label"].map(normalize_label) == out["Prediction"]).mean()), 4)
        alerts = out[out["Prediction"] != "BENIGN"].sort_values("Confidence", ascending=False)
        cols = ["Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Prediction", "Confidence"]
        if has_truth:
            cols.insert(-2, "Label")
        return {
            "rows": len(out), "dropped_rows": int(len(df) - len(out)),
            "counts": dict(counts.most_common()), "attack_rows": int(len(alerts)),
            "accuracy_vs_labels": acc,
            "alerts": alerts[cols].head(50).round(2).to_dict(orient="records"),
        }

    @app.after_request
    def headers(resp):
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        return resp

    @app.route("/", methods=["GET", "POST"])
    def index():
        result, error = None, None
        if request.method == "POST":
            try:
                if request.form.get("sample"):
                    df = pd.read_csv(SAMPLE)
                else:
                    f = request.files.get("file")
                    if not f or not f.filename.lower().endswith(".csv"):
                        raise ValueError("Upload a .csv file in CICIDS2017 MachineLearningCSV format")
                    df = pd.read_csv(io.BytesIO(f.read()), low_memory=False, encoding="latin-1")
                result = analyse(df)
            except Exception as exc:
                error = str(exc)
        return render_template("index.html", result=result, error=error, metrics=model.metrics, labels=model.labels)

    @app.post("/api/predict")
    def api_predict():
        """JSON: {"flows": [{feature: value, ...}, ...]} or CSV upload in 'file'."""
        if request.files.get("file"):
            df = pd.read_csv(request.files["file"], low_memory=False, encoding="latin-1")
        else:
            flows = (request.get_json(silent=True) or {}).get("flows")
            if not isinstance(flows, list) or not flows:
                return jsonify(error="Provide flows (list of objects) or a CSV file"), 400
            df = pd.DataFrame(flows)
        try:
            return jsonify(analyse(df))
        except ValueError as exc:
            return jsonify(error=str(exc)), 400

    @app.get("/artifacts/confusion_matrix.png")
    def cm_png():
        return send_from_directory(ARTIFACTS, "confusion_matrix.png")

    @app.get("/sample.csv")
    def sample():
        return send_from_directory(os.path.dirname(SAMPLE), "sample_flows.csv", as_attachment=True)

    return app
