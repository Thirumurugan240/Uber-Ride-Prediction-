"""Central configuration and shared constants for the project.

Keeping paths and feature definitions in one place means the training
pipeline, the Flask app and the tests all agree on the same contract.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = ROOT_DIR / "taxi.csv"
MODEL_PATH: Path = ROOT_DIR / "model.pkl"
METRICS_PATH: Path = ROOT_DIR / "metrics.json"

# --- Modelling contract -----------------------------------------------------
# Order matters: the Flask form and the model pipeline rely on this exact
# ordering of input features.
FEATURE_COLUMNS: list[str] = [
    "Priceperweek",
    "Population",
    "Monthlyincome",
    "Averageparkingpermonth",
]
TARGET_COLUMN: str = "Numberofweeklyriders"

# Human-friendly labels + validation bounds used by the web form / API.
# Bounds are intentionally generous; they exist to reject nonsense input
# (negatives, typos) rather than to enforce the training data range.
FEATURE_META: dict[str, dict] = {
    "Priceperweek": {
        "label": "Price per week ($)",
        "placeholder": "e.g. 25",
        "min": 0,
        "max": 1000,
    },
    "Population": {
        "label": "City population",
        "placeholder": "e.g. 1800000",
        "min": 0,
        "max": 100_000_000,
    },
    "Monthlyincome": {
        "label": "Average monthly income ($)",
        "placeholder": "e.g. 12000",
        "min": 0,
        "max": 10_000_000,
    },
    "Averageparkingpermonth": {
        "label": "Average parking per month ($)",
        "placeholder": "e.g. 90",
        "min": 0,
        "max": 100_000,
    },
}

RANDOM_STATE: int = 42
