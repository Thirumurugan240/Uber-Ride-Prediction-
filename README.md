# 🚖 Uber Ride Prediction

A machine-learning web application that predicts the **number of weekly Uber/transit riders** for a city from four socio-economic indicators. Built with **scikit-learn** (model) and **Flask** (web app + JSON API).

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.12+-blue.svg">
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.6+-orange.svg">
  <img alt="Flask" src="https://img.shields.io/badge/flask-3.1+-black.svg">
  <img alt="Tests" src="https://img.shields.io/badge/tests-14%20passing-brightgreen.svg">
</p>

---

## ✨ What's new in v1.0

This release upgrades the original prototype into a production-shaped project:

| Area | Before | Now |
|---|---|---|
| **Model** | Single `LinearRegression`, pickled with an old scikit-learn version (broke on load) | Automated comparison of 4 models with repeated cross-validation; best (**Gradient Boosting**) selected and persisted as a self-contained `Pipeline` |
| **Training** | Empty notebook | Reproducible `python -m src.train` pipeline + a documented EDA notebook |
| **Web app** | One route, no validation, `int()` crashes on bad input | App factory, input validation, error handling, result card |
| **API** | None | `POST /api/predict`, `GET /metrics`, `GET /health` JSON endpoints |
| **UI** | Static form | Responsive, data-driven form with a live model-metrics panel |
| **Quality** | No tests | 14 pytest unit + integration tests |
| **Ops** | None | `Dockerfile`, `requirements.txt`, gunicorn entry point |

---

## 📊 The problem

Given a city's:

| Feature | Description |
|---|---|
| `Priceperweek` | Weekly transit ticket price ($) |
| `Population` | City population |
| `Monthlyincome` | Average monthly income ($) |
| `Averageparkingpermonth` | Average monthly parking cost ($) |

…predict **`Numberofweeklyriders`** — the weekly ridership.

The model is trained on `taxi.csv` (27 rows). Because the dataset is small, models are evaluated with **repeated 5-fold cross-validation (10 repeats)** for stable estimates.

### Model leaderboard (cross-validated R²)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| **Gradient Boosting** ⭐ | **0.882** | 5,167 | 3,568 |
| Random Forest | 0.848 | 5,962 | 4,522 |
| Ridge | 0.825 | 6,295 | 5,151 |
| Linear Regression | 0.819 | 6,115 | 5,109 |

> Exact numbers are written to `metrics.json` on every training run and surfaced live in the web UI and at `GET /metrics`.

---

## 🚀 Quick start

### Option A — Poetry (recommended)

```bash
poetry install
poetry run python -m src.train      # trains model.pkl + metrics.json
poetry run python app.py            # http://localhost:5000
```

### Option B — pip / venv

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
python -m src.train
python app.py
```

Then open **http://localhost:5000**.

### Option C — Docker

```bash
docker build -t uber-ride-prediction .
docker run -p 5000:5000 uber-ride-prediction
```

---

## 🔌 API reference

### `POST /api/predict`

Predict ridership from a JSON body.

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Priceperweek": 25, "Population": 1800000, "Monthlyincome": 12000, "Averageparkingpermonth": 90}'
```

```json
{
  "prediction": 172981,
  "unit": "weekly riders",
  "model": "GradientBoosting",
  "model_version": "1.0.0"
}
```

Invalid input returns `400` with a descriptive message:

```json
{ "error": "'Price per week ($)' must be between 0 and 1,000." }
```

### `GET /metrics`

Returns the trained model's evaluation report (`metrics.json`).

### `GET /health`

Liveness/readiness probe:

```json
{ "status": "ok", "model_loaded": true, "version": "1.0.0" }
```

---

## 🧪 Testing

```bash
pytest            # or: poetry run pytest
```

Covers the prediction service (validation, bounds, type coercion, learned-relationship sanity check) and every Flask route (HTML form + JSON API + health/metrics).

---

## 🗂️ Project structure

```
Uber-Ride-Prediction-/
├── app.py                # Flask app factory: HTML form + JSON API
├── src/
│   ├── config.py         # Paths, feature contract, validation bounds
│   ├── train.py          # Train/compare/select/persist the model
│   └── predictor.py      # Loads pipeline; validates + predicts
├── templates/index.html  # Data-driven prediction UI
├── static/style.css      # Responsive styling
├── tests/                # pytest unit + integration tests
├── MLmodel.ipynb         # EDA + modelling notebook
├── taxi.csv              # Training data
├── model.pkl             # Trained scikit-learn Pipeline (generated)
├── metrics.json          # Model evaluation report (generated)
├── Dockerfile            # Container build (gunicorn)
├── requirements.txt      # pip dependencies
└── pyproject.toml        # Poetry project + pytest config
```

---

## 🔁 Retraining

Whenever `taxi.csv` changes, regenerate the model and metrics:

```bash
python -m src.train
```

This re-runs the cross-validated comparison, refits the winning model on all data, and overwrites `model.pkl` + `metrics.json`. The Flask app loads whatever model is present at startup.

---

## ⚠️ Notes & limitations

- The dataset is small (27 rows). Treat predictions as illustrative; collect more data before relying on them.
- `model.pkl` is regenerated from source, so the previous scikit-learn version-mismatch error no longer applies.
- For real deployments, run behind gunicorn (`gunicorn app:app`) — `app.run(debug=True)` is for local development only.

---

## 📜 License

Released for educational use. Author: **Thirumurgan**.
