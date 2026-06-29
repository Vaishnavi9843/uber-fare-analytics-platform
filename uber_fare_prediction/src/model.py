"""
model.py

Functions for training, saving, and loading machine learning models.
"""

import joblib

from sklearn.ensemble import RandomForestRegressor

from src.config import MODEL_PATH, FEATURE_COLUMNS_PATH


def train_random_forest(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=20,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
):
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
    feature_columns,
    model_path=MODEL_PATH,
    feature_path=FEATURE_COLUMNS_PATH,
):
    """
    Save the trained model and the feature column order.
    """

    joblib.dump(model, model_path)
    joblib.dump(feature_columns, feature_path)

    print(f"Model saved to:\n{model_path}")
    print(f"Feature columns saved to:\n{feature_path}")


def load_model(path=MODEL_PATH):
    """
    Load a trained model.

    Parameters
    ----------
    path : Path

    Returns
    -------
    sklearn model
    """

    model = joblib.load(path)

    print(f"Model loaded from:\n{path}")

    return model