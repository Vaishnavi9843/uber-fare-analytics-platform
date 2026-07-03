from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE
from src.validators import validate_required_columns


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""

    return pd.read_csv(path)


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target."""

    required_columns = ["fare_amount", "pickup_datetime"]
    validate_required_columns(df, required_columns)

    drop_columns = [
        "key",
        "pickup_datetime",
        "pickup_day",
    ]

    X = df.drop(columns=drop_columns + ["fare_amount"])
    y = df["fare_amount"]

    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the feature columns used for model training."""

    X, _ = prepare_features(df)
    return X.columns.tolist()

