"""predict.py

End-to-end fare prediction pipeline.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd

from src.config import FEATURE_DATA_PATH
from src.feature_engineering import engineer_features
from src.preprocessing import get_feature_columns


@lru_cache(maxsize=1)
def _get_feature_columns() -> list[str]:
    processed_df = pd.read_csv(FEATURE_DATA_PATH)
    return get_feature_columns(processed_df)


def prepare_prediction_data(input_data: dict[str, Any]) -> pd.DataFrame:
    """Convert user input into a DataFrame and engineer features."""

    df = pd.DataFrame([input_data])
    df = engineer_features(df)

    feature_columns = _get_feature_columns()

    # Keep only training features and preserve order
    X = df.reindex(columns=feature_columns, fill_value=0)
    return X


def predict_fare(model: Any, input_data: dict[str, Any]) -> float:
    """Predict Uber fare."""

    X = prepare_prediction_data(input_data)
    prediction = model.predict(X)
    return round(float(prediction[0]), 2)

