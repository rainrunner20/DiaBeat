"""Input and output checks shared by the demo API and isolated unit tests."""

import math

FEATURES = (
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
)


def parse_inputs(form):
    values = []
    for index, name in enumerate(FEATURES, start=1):
        raw = form.get(str(index))
        try:
            value = float(raw)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{name} must be a number.") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
        values.append(value)
    if values[-1] <= 0:
        raise ValueError("Age must be positive.")
    return values


def percent(probability):
    value = float(probability)
    if not math.isfinite(value) or not (0 <= value <= 1):
        raise ValueError("The model returned an invalid probability.")
    return f"{value * 100:.1f}%"
