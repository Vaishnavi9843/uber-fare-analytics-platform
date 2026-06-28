"""
predict.py

End-to-end fare prediction pipeline.
"""

import pandas as pd

from src.feature_engineering import engineer_features


def prepare_prediction_data(input_data):
    """
    Convert user input into a DataFrame and engineer features.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    pd.DataFrame
    """

    df = pd.DataFrame([input_data])

    df = engineer_features(df)

    drop_columns = [
        "key",
        "pickup_datetime",
        "pickup_day",
    ]

    X = df.drop(
        columns=drop_columns,
        errors="ignore",
    )

    return X


def predict_fare(model, input_data):
    """
    Predict Uber fare.

    Parameters
    ----------
    model : sklearn model

    input_data : dict

    Returns
    -------
    float
    """

    X = prepare_prediction_data(input_data)

    prediction = model.predict(X)

    return round(float(prediction[0]), 2)