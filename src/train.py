"""Train, evaluate and persist the Uber ride-prediction model.

This script replaces the original notebook-trained ``model.pkl`` (which was
pickled with scikit-learn 1.5.1 and broke on newer versions). It:

1. Loads ``taxi.csv``.
2. Builds a set of candidate regression pipelines (scaler + estimator).
3. Compares them with repeated K-fold cross-validation.
4. Refits the winner on the full dataset.
5. Persists the fitted ``Pipeline`` to ``model.pkl`` and writes a
   ``metrics.json`` report that the Flask app surfaces to users.

Run it with::

    python -m src.train
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import __version__
from src import config


@dataclass
class CandidateResult:
    """Cross-validation summary for a single candidate pipeline."""

    name: str
    pipeline: Pipeline
    r2_mean: float
    r2_std: float
    rmse_mean: float
    mae_mean: float


def load_dataset(path=config.DATA_PATH):
    """Load the taxi dataset and split into (features, target)."""
    df = pd.read_csv(path)
    missing = set(config.FEATURE_COLUMNS + [config.TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    X = df[config.FEATURE_COLUMNS].copy()
    y = df[config.TARGET_COLUMN].copy()
    return X, y


def build_candidates() -> dict[str, Pipeline]:
    """Return the candidate model pipelines to compare.

    Every candidate standard-scales its inputs first so that the linear
    models and the distance-insensitive tree models are evaluated on equal
    footing and the persisted artifact is fully self-contained.
    """
    rs = config.RANDOM_STATE
    return {
        "LinearRegression": Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "Ridge": Pipeline(
            [("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=rs))]
        ),
        "RandomForest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300, max_depth=None, random_state=rs
                    ),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=200, learning_rate=0.05, random_state=rs
                    ),
                ),
            ]
        ),
    }


def evaluate_candidates(X: pd.DataFrame, y: pd.Series) -> list[CandidateResult]:
    """Cross-validate every candidate and return their scored results.

    The dataset is tiny (~27 rows), so we use repeated K-fold to get a more
    stable estimate of generalisation performance than a single split would.
    """
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=config.RANDOM_STATE)
    scoring = {
        "r2": "r2",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
    }

    results: list[CandidateResult] = []
    for name, pipeline in build_candidates().items():
        scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring)
        results.append(
            CandidateResult(
                name=name,
                pipeline=pipeline,
                r2_mean=float(np.mean(scores["test_r2"])),
                r2_std=float(np.std(scores["test_r2"])),
                rmse_mean=float(-np.mean(scores["test_rmse"])),
                mae_mean=float(-np.mean(scores["test_mae"])),
            )
        )

    # Best = highest mean cross-validated R².
    results.sort(key=lambda r: r.r2_mean, reverse=True)
    return results


def train() -> CandidateResult:
    """Run the full training pipeline and persist the winning model."""
    X, y = load_dataset()
    results = evaluate_candidates(X, y)
    best = results[0]

    print("Cross-validation results (sorted by mean R²):")
    print(f"{'Model':<18}{'R²':>8}{'±std':>8}{'RMSE':>12}{'MAE':>12}")
    for r in results:
        print(
            f"{r.name:<18}{r.r2_mean:>8.3f}{r.r2_std:>8.3f}"
            f"{r.rmse_mean:>12.0f}{r.mae_mean:>12.0f}"
        )
    print(f"\nSelected model: {best.name}")

    # Refit the winner on ALL data before shipping it.
    best.pipeline.fit(X, y)
    joblib.dump(best.pipeline, config.MODEL_PATH)
    print(f"Saved fitted pipeline -> {config.MODEL_PATH}")

    metrics = {
        "model_version": __version__,
        "selected_model": best.name,
        "n_samples": int(len(X)),
        "features": config.FEATURE_COLUMNS,
        "target": config.TARGET_COLUMN,
        "cv": {"strategy": "RepeatedKFold", "n_splits": 5, "n_repeats": 10},
        "metrics": {
            "r2_mean": round(best.r2_mean, 4),
            "r2_std": round(best.r2_std, 4),
            "rmse_mean": round(best.rmse_mean, 2),
            "mae_mean": round(best.mae_mean, 2),
        },
        "leaderboard": [
            {
                "model": r.name,
                "r2_mean": round(r.r2_mean, 4),
                "rmse_mean": round(r.rmse_mean, 2),
                "mae_mean": round(r.mae_mean, 2),
            }
            for r in results
        ],
    }
    config.METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved metrics -> {config.METRICS_PATH}")
    return best


if __name__ == "__main__":
    train()
