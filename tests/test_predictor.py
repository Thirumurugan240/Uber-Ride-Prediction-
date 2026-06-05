"""Unit tests for the prediction service."""

import pytest

from src.predictor import Predictor, ValidationError, get_predictor

VALID = {
    "Priceperweek": 25,
    "Population": 1_800_000,
    "Monthlyincome": 12_000,
    "Averageparkingpermonth": 90,
}


@pytest.fixture(scope="module")
def predictor() -> Predictor:
    return get_predictor()


def test_predict_returns_positive_int(predictor):
    result = predictor.predict(VALID)
    assert isinstance(result, int)
    assert result > 0


def test_predict_accepts_string_values(predictor):
    # The HTML form submits everything as strings.
    string_input = {k: str(v) for k, v in VALID.items()}
    assert predictor.predict(string_input) == predictor.predict(VALID)


def test_missing_feature_raises(predictor):
    payload = dict(VALID)
    del payload["Population"]
    with pytest.raises(ValidationError, match="required"):
        predictor.predict(payload)


def test_non_numeric_raises(predictor):
    payload = dict(VALID, Priceperweek="not-a-number")
    with pytest.raises(ValidationError, match="must be a number"):
        predictor.predict(payload)


def test_out_of_bounds_raises(predictor):
    payload = dict(VALID, Priceperweek=-10)
    with pytest.raises(ValidationError, match="between"):
        predictor.predict(payload)


def test_higher_price_lowers_ridership(predictor):
    """Sanity check the learned relationship: price is negatively correlated."""
    cheap = predictor.predict(dict(VALID, Priceperweek=15))
    pricey = predictor.predict(dict(VALID, Priceperweek=100))
    assert cheap >= pricey
