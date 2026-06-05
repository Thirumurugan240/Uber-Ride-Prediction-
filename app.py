"""Flask web app + JSON API for Uber weekly-ride prediction.

Routes
------
GET  /              HTML form (renders the prediction UI).
POST /predict       Handles the HTML form submission, re-renders with a result.
POST /api/predict   JSON API: {"Priceperweek": 25, ...} -> {"prediction": 12345}.
GET  /metrics       Returns the trained model's evaluation metrics as JSON.
GET  /health        Liveness/readiness probe.
"""

from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request

from src import __version__, config
from src.predictor import ValidationError, get_predictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Application factory — builds and configures the Flask app."""
    app = Flask(__name__)

    # Fail fast at startup if the model is missing/corrupt.
    predictor = get_predictor()
    logger.info("Loaded model: %s", predictor.metrics.get("selected_model", "unknown"))

    @app.route("/")
    def home():
        return render_template(
            "index.html",
            features=config.FEATURE_META,
            metrics=predictor.metrics,
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        try:
            prediction = predictor.predict(request.form.to_dict())
            result = {
                "value": prediction,
                "text": f"Estimated weekly riders: {prediction:,}",
            }
            return render_template(
                "index.html",
                features=config.FEATURE_META,
                metrics=predictor.metrics,
                result=result,
                submitted=request.form.to_dict(),
            )
        except ValidationError as exc:
            return (
                render_template(
                    "index.html",
                    features=config.FEATURE_META,
                    metrics=predictor.metrics,
                    error=str(exc),
                    submitted=request.form.to_dict(),
                ),
                400,
            )

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        payload = request.get_json(silent=True)
        if payload is None:
            # Fall back to form-encoded bodies for convenience.
            payload = request.form.to_dict()
        if not payload:
            return jsonify(error="Request body must be JSON with the feature fields."), 400
        try:
            prediction = predictor.predict(payload)
        except ValidationError as exc:
            return jsonify(error=str(exc)), 400
        return jsonify(
            prediction=prediction,
            unit="weekly riders",
            model=predictor.metrics.get("selected_model"),
            model_version=__version__,
        )

    @app.route("/metrics")
    def metrics():
        return jsonify(predictor.metrics)

    @app.route("/health")
    def health():
        return jsonify(status="ok", model_loaded=True, version=__version__)

    return app


# WSGI entry point (used by gunicorn: `gunicorn app:app`).
app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
