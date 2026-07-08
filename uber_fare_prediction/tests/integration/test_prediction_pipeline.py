"""
tests/integration/test_prediction_pipeline.py

Integration tests for the prediction pipeline.
"""

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.predict import (
    prepare_prediction_data,
    predict_fare,
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def sample_input():
    return {
        "pickup_datetime": "2015-05-02 08:30:00",
        "pickup_longitude": -73.985428,
        "pickup_latitude": 40.748817,
        "dropoff_longitude": -73.985130,
        "dropoff_latitude": 40.758896,
        "passenger_count": 2,
    }


@pytest.fixture
def trained_model():

    X = pd.DataFrame(
        {
            "pickup_hour": [8, 10, 12, 15],
            "trip_distance_km": [2.1, 5.3, 1.4, 7.2],
            "passenger_count": [1, 2, 3, 1],
            "is_weekend": [1, 0, 0, 1],
            "is_peak_hour": [1, 0, 0, 1],
        }
    )

    y = pd.Series(
        [10.2, 18.4, 7.9, 23.1]
    )

    model = RandomForestRegressor(
        random_state=42,
        n_estimators=10,
    )

    model.fit(X, y)

    return model


# ==========================================================
# Full Prediction Pipeline
# ==========================================================

def test_prediction_pipeline(
    sample_input,
    trained_model,
    monkeypatch,
):

    monkeypatch.setattr(
        "src.predict._get_feature_columns",
        lambda: [
            "pickup_hour",
            "trip_distance_km",
            "passenger_count",
            "is_weekend",
            "is_peak_hour",
        ],
    )

    prediction = predict_fare(
        trained_model,
        sample_input,
    )

    assert isinstance(
        prediction,
        float,
    )


def test_prepare_prediction_pipeline(
    sample_input,
    monkeypatch,
):

    monkeypatch.setattr(
        "src.predict._get_feature_columns",
        lambda: [
            "pickup_hour",
            "trip_distance_km",
            "passenger_count",
            "is_weekend",
            "is_peak_hour",
        ],
    )

    X = prepare_prediction_data(sample_input)

    assert isinstance(
        X,
        pd.DataFrame,
    )

    assert len(X) == 1


def test_prediction_not_negative(
    sample_input,
    trained_model,
    monkeypatch,
):

    monkeypatch.setattr(
        "src.predict._get_feature_columns",
        lambda: [
            "pickup_hour",
            "trip_distance_km",
            "passenger_count",
            "is_weekend",
            "is_peak_hour",
        ],
    )

    prediction = predict_fare(
        trained_model,
        sample_input,
    )

    assert prediction >= 0


def test_prediction_consistency(
    sample_input,
    trained_model,
    monkeypatch,
):

    monkeypatch.setattr(
        "src.predict._get_feature_columns",
        lambda: [
            "pickup_hour",
            "trip_distance_km",
            "passenger_count",
            "is_weekend",
            "is_peak_hour",
        ],
    )

    prediction1 = predict_fare(
        trained_model,
        sample_input,
    )

    prediction2 = predict_fare(
        trained_model,
        sample_input,
    )

    assert prediction1 == prediction2