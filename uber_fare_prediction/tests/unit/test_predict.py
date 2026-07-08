"""
tests/unit/test_predict.py

Unit tests for prediction pipeline.
"""

import pandas as pd
import pytest

from src.predict import (
    prepare_prediction_data,
    predict_fare,
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def sample_input():
    """
    Sample prediction request.
    """

    return {
        "pickup_datetime": "2015-05-02 08:30:00",
        "pickup_longitude": -73.985428,
        "pickup_latitude": 40.748817,
        "dropoff_longitude": -73.985130,
        "dropoff_latitude": 40.758896,
        "passenger_count": 2,
    }


@pytest.fixture
def expected_columns():
    """
    Feature columns used during training.
    """

    return [
        "pickup_hour",
        "trip_distance_km",
        "passenger_count",
        "is_weekend",
        "is_peak_hour",
    ]


# ==========================================================
# Fake Model
# ==========================================================

class FakeModel:
    """
    Mock model used for testing.
    """

    def predict(self, X):
        return [18.456]


# ==========================================================
# prepare_prediction_data()
# ==========================================================

def test_prepare_prediction_data(
    sample_input,
    expected_columns,
    monkeypatch,
):
    """
    Test prediction feature preparation.
    """

    def fake_engineer_features(df):
        df = df.copy()

        df["pickup_hour"] = 8
        df["trip_distance_km"] = 2.5
        df["is_weekend"] = 1
        df["is_peak_hour"] = 1

        return df

    monkeypatch.setattr(
        "src.predict.engineer_features",
        fake_engineer_features,
    )

    monkeypatch.setattr(
        "src.predict._get_feature_columns",
        lambda: expected_columns,
    )

    X = prepare_prediction_data(sample_input)

    assert isinstance(X, pd.DataFrame)

    assert list(X.columns) == expected_columns

    assert len(X) == 1


def test_prepare_prediction_data_preserves_column_order(
    sample_input,
    expected_columns,
    monkeypatch,
):
    """
    Column order must match training.
    """

    def fake_engineer_features(df):
        df = df.copy()

        df["trip_distance_km"] = 2.5
        df["pickup_hour"] = 8
        df["is_peak_hour"] = 1
        df["is_weekend"] = 1

        return df

    monkeypatch.setattr(
        "src.predict.engineer_features",
        fake_engineer_features,
    )

    monkeypatch.setattr(
        "src.predict._get_feature_columns",
        lambda: expected_columns,
    )

    X = prepare_prediction_data(sample_input)

    assert list(X.columns) == expected_columns


def test_prepare_prediction_data_returns_dataframe(
    sample_input,
    expected_columns,
    monkeypatch,
):
    """
    Output should be DataFrame.
    """

    monkeypatch.setattr(
        "src.predict.engineer_features",
        lambda df: df,
    )

    monkeypatch.setattr(
        "src.predict._get_feature_columns",
        lambda: expected_columns,
    )

    X = prepare_prediction_data(sample_input)

    assert isinstance(X, pd.DataFrame)


# ==========================================================
# predict_fare()
# ==========================================================

def test_predict_fare_returns_float(
    sample_input,
    expected_columns,
    monkeypatch,
):
    """
    Prediction should be float.
    """

    def fake_engineer_features(df):
        df = df.copy()

        df["pickup_hour"] = 8
        df["trip_distance_km"] = 2.5
        df["is_weekend"] = 1
        df["is_peak_hour"] = 1

        return df

    monkeypatch.setattr(
        "src.predict.engineer_features",
        fake_engineer_features,
    )

    monkeypatch.setattr(
        "src.predict._get_feature_columns",
        lambda: expected_columns,
    )

    prediction = predict_fare(
        FakeModel(),
        sample_input,
    )

    assert isinstance(prediction, float)


def test_predict_fare_rounding(
    sample_input,
    expected_columns,
    monkeypatch,
):
    """
    Prediction should be rounded to 2 decimals.
    """

    def fake_engineer_features(df):
        df = df.copy()

        df["pickup_hour"] = 8
        df["trip_distance_km"] = 2.5
        df["is_weekend"] = 1
        df["is_peak_hour"] = 1

        return df

    monkeypatch.setattr(
        "src.predict.engineer_features",
        fake_engineer_features,
    )

    monkeypatch.setattr(
        "src.predict._get_feature_columns",
        lambda: expected_columns,
    )

    prediction = predict_fare(
        FakeModel(),
        sample_input,
    )

    assert prediction == 18.46


def test_predict_fare_calls_model(
    sample_input,
    expected_columns,
    monkeypatch,
):
    """
    Ensure model.predict() is called.
    """

    called = False

    class MockModel:

        def predict(self, X):
            nonlocal called
            called = True
            return [25.0]

    def fake_engineer_features(df):
        df = df.copy()

        df["pickup_hour"] = 8
        df["trip_distance_km"] = 2.5
        df["is_weekend"] = 1
        df["is_peak_hour"] = 1

        return df

    monkeypatch.setattr(
        "src.predict.engineer_features",
        fake_engineer_features,
    )

    monkeypatch.setattr(
        "src.predict._get_feature_columns",
        lambda: expected_columns,
    )

    predict_fare(
        MockModel(),
        sample_input,
    )

    assert called


# ==========================================================
# Edge Cases
# ==========================================================

def test_prepare_prediction_data_missing_columns(
    sample_input,
    monkeypatch,
):
    """
    Missing engineered columns should be filled with zero.
    """

    expected_columns = [
        "pickup_hour",
        "trip_distance_km",
        "passenger_count",
        "is_weekend",
        "is_peak_hour",
    ]

    monkeypatch.setattr(
        "src.predict.engineer_features",
        lambda df: df,
    )

    monkeypatch.setattr(
        "src.predict._get_feature_columns",
        lambda: expected_columns,
    )

    X = prepare_prediction_data(sample_input)

    assert "pickup_hour" in X.columns

    assert X.loc[0, "pickup_hour"] == 0

    assert X.loc[0, "trip_distance_km"] == 0