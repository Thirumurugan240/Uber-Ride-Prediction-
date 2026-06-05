"""Integration tests for the Flask routes."""

import pytest

from app import create_app

VALID = {
    "Priceperweek": 25,
    "Population": 1_800_000,
    "Monthlyincome": 12_000,
    "Averageparkingpermonth": 90,
}


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def test_home_renders(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"Uber Rides Predictor" in resp.data


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.get_json()["status"] == "ok"


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "selected_model" in resp.get_json()


def test_api_predict_valid(client):
    resp = client.post("/api/predict", json=VALID)
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["prediction"] > 0
    assert body["unit"] == "weekly riders"


def test_api_predict_invalid(client):
    resp = client.post("/api/predict", json=dict(VALID, Population="oops"))
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_api_predict_empty_body(client):
    resp = client.post("/api/predict", json={})
    assert resp.status_code == 400


def test_form_predict_renders_result(client):
    resp = client.post("/predict", data=VALID)
    assert resp.status_code == 200
    assert b"Estimated weekly riders" in resp.data


def test_form_predict_shows_error(client):
    resp = client.post("/predict", data=dict(VALID, Priceperweek="abc"))
    assert resp.status_code == 400
    assert b"alert-error" in resp.data
