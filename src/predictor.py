"""Prediction service: loads the trained pipeline and validates input.

Separating this from the Flask layer keeps the model logic testable without
spinning up a web server, and gives both the HTML form and the JSON API a
single, consistent validation + inference path.
"""

from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd

from src import config


class ValidationError(ValueError):
    """Raised when user-supplied features fail validation."""


class Predictor:
    """Thin wrapper around the persisted scikit-learn pipeline."""

    def __init__(self, model_path=config.MODEL_PATH, metrics_path=config.METRICS_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {model_path}. "
                "Train it first with: python -m src.train"
            )
        self.model = joblib.load(model_path)
        self.metrics = (
            json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
        )

    def validate(self, raw: dict) -> dict[str, float]:
        """Validate and coerce a raw feature mapping to floats.

        Accepts string or numeric values (as the HTML form / JSON API send),
        checks every required feature is present, numeric and within bounds.
        """
        cleaned: dict[str, float] = {}
        errors: list[str] = []

        for feature in config.FEATURE_COLUMNS:
            meta = config.FEATURE_META[feature]
            if feature not in raw or raw[feature] in (None, ""):
                errors.append(f"'{meta['label']}' is required.")
                continue
            try:
                value = float(raw[feature])
            except (TypeError, ValueError):
                errors.append(f"'{meta['label']}' must be a number.")
                continue
            if value < meta["min"] or value > meta["max"]:
                errors.append(
                    f"'{meta['label']}' must be between {meta['min']:,} "
                    f"and {meta['max']:,}."
                )
                continue
            cleaned[feature] = value

        if errors:
            raise ValidationError(" ".join(errors))
        return cleaned

    def predict(self, raw: dict) -> int:
        """Validate input and return the predicted weekly rider count."""
        cleaned = self.validate(raw)
        # Predict with a DataFrame so feature names match training and
        # scikit-learn does not emit a "missing feature names" warning.
        frame = pd.DataFrame([cleaned], columns=config.FEATURE_COLUMNS)
        prediction = self.model.predict(frame)[0]
        return max(0, int(round(float(prediction))))


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Return a process-wide cached Predictor (model loaded once)."""
    return Predictor()
