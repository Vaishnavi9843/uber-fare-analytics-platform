"""
Prediction service.

Loads the trained model once and performs predictions.
"""

from functools import lru_cache

from src.model import load_model
from src.predict import predict_fare


@lru_cache(maxsize=1)
def get_model():
    """
    Load the trained model once.
    """
    return load_model()


def make_prediction(input_data: dict) -> float:
    """
    Predict Uber fare.

    Parameters
    ----------
    input_data : dict

    Returns
    -------
    float
    """

    model = get_model()

    return predict_fare(
        model=model,
        input_data=input_data,
    )