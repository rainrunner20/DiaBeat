"""Educational diabetes classification demo, not clinical decision support."""

import os
import pickle
from functools import lru_cache
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request
from flask_cors import CORS

from model_contract import FEATURES, parse_inputs, percent

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
CORS(app, resources={"/predict": {
    "origins": [value.strip() for value in
                os.getenv("DIABEAT_ALLOWED_ORIGINS", "http://localhost:3000").split(",")
                if value.strip()]
}})


@lru_cache(maxsize=1)
def get_models():
    # Pickles can execute code on load. Use ONLY model artifacts included
    # in this trusted repository, never uploaded or third-party pickle files.
    result = {}
    for key, filename in [
        ("KNN", "knn_model.pkl"),
        ("XGBoost", "xgb_model.pkl"),
        ("Logistic Regression", "logreg_model.pkl"),
        ("Random Forest", "rf_model.pkl"),
        ("scaler", "scaler.pkl"),
    ]:
        with (ROOT / filename).open("rb") as saved:
            result[key] = pickle.load(saved)
    return result


@app.get("/health")
def health():
    return {"status": "ok", "model_loading": "lazy", "clinical_use": False}


@app.get("/")
def template_deploy():
    return render_template("index.html")


@app.post("/predict")
def predict():
    try:
        values = parse_inputs(request.form)
    except ValueError as error:
        return render_template("result.html", error=str(error), results=None), 400

    try:
        models = get_models()
        frame = pd.DataFrame([values], columns=FEATURES)
        scaled = models["scaler"].transform(frame)
        outputs = []
        for name in ("KNN", "XGBoost", "Logistic Regression", "Random Forest"):
            features = frame if name == "XGBoost" else scaled
            probability = models[name].predict_proba(features)[0][1]
            outputs.append({"name": name, "value": percent(probability)})
        return render_template("result.html", results=outputs, error=None)
    except Exception:
        # Do not log submitted medical measurements or expose model internals.
        app.logger.exception("Model inference unavailable")
        return render_template(
            "result.html", results=None,
            error="The research model could not evaluate this example."
        ), 503


if __name__ == "__main__":
    app.run(debug=False)
