"""
predict.py

End-to-end fare prediction pipeline.
"""

import pandas as pd

from src.config import FEATURE_DATA_PATH
from src.feature_engineering import engineer_features
from src.preprocessing import get_feature_columns


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

    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_data])

    # Apply feature engineering
    df = engineer_features(df)

    # Load processed dataset to retrieve feature order
    processed_df = pd.read_csv(FEATURE_DATA_PATH)

    feature_columns = get_feature_columns(processed_df)

    # Keep only training features and preserve order
    X = df.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    return X


def predict_fare(model, input_data):
    """
    Predict Uber fare.

    Parameters
    ----------
    model : sklearn estimator

    input_data : dict

    Returns
    -------
    float
    """

    X = prepare_prediction_data(input_data)

    prediction = model.predict(X)

    return round(float(prediction[0]), 2)