"""
tests/unit/test_preprocessing.py

Unit tests for preprocessing functions.
"""

from pathlib import Path

import pandas as pd
import pytest

from src.preprocessing import (
    get_feature_columns,
    load_dataset,
    prepare_features,
    split_dataset,
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def sample_dataframe():
    """
    Sample feature-engineered dataset.
    """

    return pd.DataFrame(
        {
            "key": ["abc123", "def456", "ghi789"],
            "fare_amount": [10.5, 18.2, 7.8],
            "pickup_datetime": [
                "2015-05-02 08:30:00",
                "2015-05-03 10:00:00",
                "2015-05-04 13:15:00",
            ],
            "pickup_day": [
                "Saturday",
                "Sunday",
                "Monday",
            ],
            "pickup_hour": [8, 10, 13],
            "trip_distance_km": [2.5, 6.1, 1.4],
            "passenger_count": [1, 2, 3],
            "is_weekend": [1, 1, 0],
            "is_peak_hour": [1, 0, 0],
        }
    )


# ==========================================================
# Dataset Loading
# ==========================================================

def test_load_dataset(tmp_path: Path, sample_dataframe):
    """
    Test CSV loading.
    """

    csv_path = tmp_path / "sample.csv"

    sample_dataframe.to_csv(csv_path, index=False)

    loaded_df = load_dataset(csv_path)

    pd.testing.assert_frame_equal(
        loaded_df,
        sample_dataframe,
    )


# ==========================================================
# Feature Preparation
# ==========================================================

def test_prepare_features(sample_dataframe):
    """
    Test feature/target separation.
    """

    X, y = prepare_features(sample_dataframe)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    assert "fare_amount" not in X.columns
    assert "pickup_datetime" not in X.columns
    assert "pickup_day" not in X.columns
    assert "key" not in X.columns

    assert len(X) == len(y)


def test_prepare_features_target_values(sample_dataframe):
    """
    Target values should remain unchanged.
    """

    _, y = prepare_features(sample_dataframe)

    expected = sample_dataframe["fare_amount"]

    pd.testing.assert_series_equal(
        y,
        expected,
    )


def test_prepare_features_missing_required_column(sample_dataframe):
    """
    Missing required columns should raise an exception.
    """

    df = sample_dataframe.drop(columns=["fare_amount"])

    with pytest.raises(Exception):
        prepare_features(df)


# ==========================================================
# Feature Columns
# ==========================================================

def test_get_feature_columns(sample_dataframe):
    """
    Test returned feature names.
    """

    columns = get_feature_columns(sample_dataframe)

    expected = [
        "pickup_hour",
        "trip_distance_km",
        "passenger_count",
        "is_weekend",
        "is_peak_hour",
    ]

    assert columns == expected


# ==========================================================
# Train/Test Split
# ==========================================================

def test_split_dataset(sample_dataframe):
    """
    Test train/test split.
    """

    X, y = prepare_features(sample_dataframe)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    assert len(X_train) + len(X_test) == len(X)

    assert len(y_train) + len(y_test) == len(y)


def test_split_dataset_returns_correct_types(sample_dataframe):
    """
    Verify return types.
    """

    X, y = prepare_features(sample_dataframe)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)

    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)


# ==========================================================
# Data Integrity
# ==========================================================

def test_prepare_features_does_not_modify_original_dataframe(
    sample_dataframe,
):
    """
    Original DataFrame should remain unchanged.
    """

    original = sample_dataframe.copy(deep=True)

    prepare_features(sample_dataframe)

    pd.testing.assert_frame_equal(
        sample_dataframe,
        original,
    )


def test_get_feature_columns_matches_prepare_features(
    sample_dataframe,
):
    """
    Feature column helper should match X.columns.
    """

    X, _ = prepare_features(sample_dataframe)

    columns = get_feature_columns(sample_dataframe)

    assert columns == X.columns.tolist()