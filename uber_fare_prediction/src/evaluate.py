"""
evaluate.py

Utility functions for evaluating regression models.
"""

import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_regression_model(y_true, y_pred):
    """
    Evaluate regression predictions.

    Parameters
    ----------
    y_true : array-like

    y_pred : array-like

    Returns
    -------
    dict
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred,
        )
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4),
    }


def print_metrics(metrics):
    """
    Print evaluation metrics.
    """

    print("-" * 40)

    for key, value in metrics.items():
        print(f"{key:<15}: {value}")

    print("-" * 40)