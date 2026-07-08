"""
tests/unit/test_feature_engineering.py

Unit tests for feature engineering functions.
"""

import pandas as pd
import pytest

from src.feature_engineering import (
    add_time_features,
    add_weekend_feature,
    add_peak_hour_feature,
    calculate_trip_distance,
    engineer_features,
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def sample_trip():
    """
    Sample trip occurring on a Saturday during peak hours.
    """

    return pd.DataFrame(
        {
            "pickup_datetime": ["2015-05-02 08:30:00"],
            "pickup_longitude": [-73.985428],
            "pickup_latitude": [40.748817],
            "dropoff_longitude": [-73.985130],
            "dropoff_latitude": [40.758896],
            "passenger_count": [2],
        }
    )


# ==========================================================
# Time Features
# ==========================================================

def test_add_time_features(sample_trip):
    """
    Test temporal feature extraction.
    """

    df = add_time_features(sample_trip)

    expected_columns = [
        "pickup_hour",
        "pickup_day",
        "day_of_week",
        "pickup_month",
        "pickup_year",
    ]

    for column in expected_columns:
        assert column in df.columns

    assert df.loc[0, "pickup_hour"] == 8
    assert df.loc[0, "pickup_month"] == 5
    assert df.loc[0, "pickup_year"] == 2015


# ==========================================================
# Weekend Feature
# ==========================================================

def test_add_weekend_feature(sample_trip):
    """
    Saturday should be marked as weekend.
    """

    df = add_time_features(sample_trip)
    df = add_weekend_feature(df)

    assert "is_weekend" in df.columns
    assert df.loc[0, "is_weekend"] == 1


def test_weekday_not_weekend():
    """
    Monday should not be marked as weekend.
    """

    df = pd.DataFrame(
        {
            "pickup_datetime": ["2015-05-04 10:00:00"],
            "pickup_longitude": [-73.98],
            "pickup_latitude": [40.75],
            "dropoff_longitude": [-73.97],
            "dropoff_latitude": [40.76],
            "passenger_count": [1],
        }
    )

    df = add_time_features(df)
    df = add_weekend_feature(df)

    assert df.loc[0, "is_weekend"] == 0


# ==========================================================
# Peak Hour Feature
# ==========================================================

def test_add_peak_hour_feature(sample_trip):
    """
    08:30 AM is a peak hour.
    """

    df = add_time_features(sample_trip)
    df = add_peak_hour_feature(df)

    assert "is_peak_hour" in df.columns
    assert df.loc[0, "is_peak_hour"] == 1


def test_non_peak_hour():
    """
    1 PM should not be considered peak hour.
    """

    df = pd.DataFrame(
        {
            "pickup_datetime": ["2015-05-04 13:00:00"],
            "pickup_longitude": [-73.98],
            "pickup_latitude": [40.75],
            "dropoff_longitude": [-73.97],
            "dropoff_latitude": [40.76],
            "passenger_count": [1],
        }
    )

    df = add_time_features(df)
    df = add_peak_hour_feature(df)

    assert df.loc[0, "is_peak_hour"] == 0


# ==========================================================
# Trip Distance
# ==========================================================

def test_calculate_trip_distance(sample_trip):
    """
    Trip distance should be positive.
    """

    df = calculate_trip_distance(sample_trip)

    assert "trip_distance_km" in df.columns
    assert df.loc[0, "trip_distance_km"] > 0


def test_zero_trip_distance():
    """
    Same pickup and dropoff coordinates should produce zero distance.
    """

    df = pd.DataFrame(
        {
            "pickup_datetime": ["2015-05-04 10:00:00"],
            "pickup_longitude": [-73.98],
            "pickup_latitude": [40.75],
            "dropoff_longitude": [-73.98],
            "dropoff_latitude": [40.75],
            "passenger_count": [1],
        }
    )

    df = calculate_trip_distance(df)

    assert df.loc[0, "trip_distance_km"] == pytest.approx(
        0.0,
        abs=1e-6,
    )


# ==========================================================
# Complete Feature Engineering Pipeline
# ==========================================================

def test_engineer_features(sample_trip):
    """
    Test complete feature engineering pipeline.
    """

    df = engineer_features(sample_trip)

    expected_columns = [
        "pickup_hour",
        "pickup_day",
        "day_of_week",
        "pickup_month",
        "pickup_year",
        "is_weekend",
        "is_peak_hour",
        "trip_distance_km",
    ]

    for column in expected_columns:
        assert column in df.columns


def test_engineer_features_returns_dataframe(sample_trip):
    """
    Pipeline should return a DataFrame.
    """

    result = engineer_features(sample_trip)

    assert isinstance(result, pd.DataFrame)


def test_engineer_features_preserves_rows(sample_trip):
    """
    Feature engineering must not change row count.
    """

    result = engineer_features(sample_trip)

    assert len(result) == len(sample_trip)


def test_original_dataframe_not_modified(sample_trip):
    """
    Original DataFrame should remain unchanged.
    """

    original = sample_trip.copy(deep=True)

    _ = engineer_features(sample_trip)

    pd.testing.assert_frame_equal(
        sample_trip,
        original,
    )


def test_no_missing_engineered_values(sample_trip):
    """
    Engineered features should not contain missing values.
    """

    result = engineer_features(sample_trip)

    engineered_columns = [
        "pickup_hour",
        "pickup_day",
        "day_of_week",
        "pickup_month",
        "pickup_year",
        "is_weekend",
        "is_peak_hour",
        "trip_distance_km",
    ]

    assert (
        result[engineered_columns]
        .isnull()
        .sum()
        .sum()
        == 0
    )