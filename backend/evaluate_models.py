"""Reproducible research benchmark on the original Pima-style CSV.

This is a held-out educational classification test, not clinical validation.
Do not use cleaned_diabetes.csv: it was already transformed outside a split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from model_contract import FEATURES

ZERO_AS_MISSING = ("Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI")


def evaluate(csv_path):
    frame = pd.read_csv(csv_path)
    expected = set(FEATURES) | {"Outcome"}
    if not expected.issubset(frame.columns):
        raise ValueError("Dataset is missing required features or Outcome.")
    if len(frame) < 100 or frame[list(expected)].isna().any().any():
        raise ValueError("The benchmark requires at least 100 complete source rows.")
    if not set(frame["Outcome"].unique()).issubset({0, 1}) or frame["Outcome"].nunique() != 2:
        raise ValueError("Outcome must contain both binary classes.")
    data = frame[list(FEATURES)].copy()
    for column in ZERO_AS_MISSING:
        data[column] = data[column].replace(0, np.nan)
    labels = frame["Outcome"]
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.25, random_state=42, stratify=labels
    )
    models = {
        "Most frequent baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": LogisticRegression(max_iter=1000, solver="liblinear", random_state=42),
        "K-nearest neighbors": KNeighborsClassifier(n_neighbors=7),
        "Random forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced", random_state=42
        ),
    }
    results = []
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", estimator),
        ])
        pipeline.fit(x_train, y_train)
        predicted = pipeline.predict(x_test)
        probabilities = pipeline.predict_proba(x_test)[:, 1]
        results.append({
            "model": name,
            "accuracy": round(float(accuracy_score(y_test, predicted)), 4),
            "f1": round(float(f1_score(y_test, predicted, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, predicted, zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        })

    return {
        "source": str(Path(csv_path).name),
        "rows": len(frame),
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "split": "stratified random holdout 75/25, seed=42",
        "zero_as_missing": list(ZERO_AS_MISSING),
        "models": results,
        "notes": (
            "All imputation and scaling fit on training data only. Single holdout; "
            "no external cohort, calibration or clinical validation. "
            "These fitted evaluation models are separate from legacy web app pickle artifacts."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate the raw Pima-style diabetes dataset")
    parser.add_argument("--input", default="backend/diabetes.csv")
    parser.add_argument("--output", default="reports/diabeat_metrics.json")
    args = parser.parse_args()
    try:
        results = evaluate(args.input)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved holdout metrics: {output}")


if __name__ == "__main__":
    main()
