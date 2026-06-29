"""
predict.py

End-to-end fare prediction pipeline.
"""

import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS_PATH
from src.feature_engineering import engineer_features


def prepare_prediction_data(input_data):
    """
    Convert user input into a DataFrame and engineer features.
    """

    df = pd.DataFrame([input_data])

    df = engineer_features(df)

    drop_columns = [
        "key",
        "fare_amount",
        "pickup_datetime",
        "pickup_day",
    ]

    X = df.drop(
        columns=drop_columns,
        errors="ignore",
    )

    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

    X = X.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    return X


def predict_fare(model, input_data):
    """
    Predict Uber fare.
    """

    X = prepare_prediction_data(input_data)

    prediction = model.predict(X)

    return round(float(prediction[0]), 2)