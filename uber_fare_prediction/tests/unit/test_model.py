"""
tests/unit/test_model.py

Unit tests for model training, saving, and loading.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from src.model import (
    load_model,
    save_model,
    train_random_forest,
)


# ==========================================================
# Fixtures
# ==========================================================

@pytest.fixture
def sample_training_data():
    """
    Small synthetic training dataset.
    """

    X = pd.DataFrame(
        {
            "pickup_hour": [8, 9, 10, 11, 12],
            "trip_distance_km": [1.2, 3.5, 5.1, 2.7, 6.3],
            "passenger_count": [1, 2, 1, 3, 2],
            "is_weekend": [0, 0, 1, 1, 0],
            "is_peak_hour": [1, 1, 0, 0, 0],
        }
    )

    y = pd.Series(
        [8.5, 12.3, 18.2, 10.1, 21.7],
        name="fare_amount",
    )

    return X, y


# ==========================================================
# Model Training
# ==========================================================

def test_train_random_forest_returns_model(
    sample_training_data,
):
    """
    Training should return a RandomForestRegressor.
    """

    X, y = sample_training_data

    model = train_random_forest(X, y)

    assert isinstance(
        model,
        RandomForestRegressor,
    )


def test_model_is_fitted(
    sample_training_data,
):
    """
    Model should be fitted after training.
    """

    X, y = sample_training_data

    model = train_random_forest(X, y)

    assert hasattr(model, "estimators_")
    assert len(model.estimators_) > 0


def test_model_can_predict(
    sample_training_data,
):
    """
    Trained model should generate predictions.
    """

    X, y = sample_training_data

    model = train_random_forest(X, y)

    prediction = model.predict(X)

    assert len(prediction) == len(X)

    assert np.issubdtype(
        prediction.dtype,
        np.number,
    )


# ==========================================================
# Model Saving
# ==========================================================

def test_save_model_creates_file(
    sample_training_data,
    tmp_path: Path,
):
    """
    Saving should create a model file.
    """

    X, y = sample_training_data

    model = train_random_forest(X, y)

    model_path = tmp_path / "model.pkl"

    save_model(
        model,
        model_path=model_path,
    )

    assert model_path.exists()


# ==========================================================
# Model Loading
# ==========================================================

def test_load_model_returns_model(
    sample_training_data,
    tmp_path: Path,
):
    """
    Loading should return RandomForestRegressor.
    """

    X, y = sample_training_data

    model = train_random_forest(X, y)

    model_path = tmp_path / "model.pkl"

    save_model(
        model,
        model_path=model_path,
    )

    loaded_model = load_model(model_path)

    assert isinstance(
        loaded_model,
        RandomForestRegressor,
    )


def test_loaded_model_predicts(
    sample_training_data,
    tmp_path: Path,
):
    """
    Loaded model should make predictions.
    """

    X, y = sample_training_data

    model = train_random_forest(X, y)

    model_path = tmp_path / "model.pkl"

    save_model(
        model,
        model_path=model_path,
    )

    loaded_model = load_model(model_path)

    prediction = loaded_model.predict(X)

    assert len(prediction) == len(X)

    assert np.issubdtype(
        prediction.dtype,
        np.number,
    )


# ==========================================================
# Persistence
# ==========================================================

def test_saved_and_loaded_predictions_match(
    sample_training_data,
    tmp_path: Path,
):
    """
    Saved and loaded model should produce identical predictions.
    """

    X, y = sample_training_data

    model = train_random_forest(X, y)

    original_prediction = model.predict(X)

    model_path = tmp_path / "model.pkl"

    save_model(
        model,
        model_path=model_path,
    )

    loaded_model = load_model(model_path)

    loaded_prediction = loaded_model.predict(X)

    np.testing.assert_allclose(
        original_prediction,
        loaded_prediction,
    )


# ==========================================================
# Error Handling
# ==========================================================

def test_load_nonexistent_model():
    """
    Loading a missing model should raise FileNotFoundError.
    """

    with pytest.raises(FileNotFoundError):
        load_model("does_not_exist.pkl")