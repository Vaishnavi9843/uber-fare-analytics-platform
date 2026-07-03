"""model.py

Functions for training, saving, and loading machine learning models.
"""

from __future__ import annotations

import joblib
from sklearn.ensemble import RandomForestRegressor

from src.config import MODEL_PATH
from src.logger import logger


def train_random_forest(
    X_train,
    y_train,
    n_estimators: int = 100,
    max_depth: int | None = 20,
    min_samples_split: int = 5,
    min_samples_leaf: int = 2,
    max_features: str | int | float = "sqrt",
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model.fit(X_train, y_train)
    return model


def save_model(
    model,
    model_path=MODEL_PATH,
) -> None:
    """Save the trained model."""

    joblib.dump(model, model_path)
    logger.info("Model saved to: %s", model_path)


def load_model(path=MODEL_PATH) -> RandomForestRegressor:
    """Load a trained model."""

    model = joblib.load(path)
    logger.info("Model loaded from: %s", path)
    return model

