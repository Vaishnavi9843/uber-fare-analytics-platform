import numpy as np
import pandas as pd
EARTH_RADIUS_KM = 6371

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal features from pickup_datetime.
    """

    df = df.copy()

    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])

    df["pickup_hour"] = df["pickup_datetime"].dt.hour

    df["pickup_day"] = df["pickup_datetime"].dt.day_name()

    df["day_of_week"] = df["pickup_datetime"].dt.dayofweek

    df["pickup_month"] = df["pickup_datetime"].dt.month

    df["pickup_year"] = df["pickup_datetime"].dt.year

    return df

def add_weekend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create weekend flag.
    """

    df = df.copy()

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    return df

def add_peak_hour_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create peak hour indicator.
    """

    df = df.copy()

    df["is_peak_hour"] = (
        (
            (df["pickup_hour"] >= 7)
            & (df["pickup_hour"] <= 9)
        )
        |
        (
            (df["pickup_hour"] >= 16)
            & (df["pickup_hour"] <= 19)
        )
    ).astype(int)

    return df

def calculate_trip_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate trip distance in kilometers using the Haversine formula.
    """

    df = df.copy()

    lat1 = np.radians(df["pickup_latitude"])
    lon1 = np.radians(df["pickup_longitude"])

    lat2 = np.radians(df["dropoff_latitude"])
    lon2 = np.radians(df["dropoff_longitude"])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arctan2(
        np.sqrt(a),
        np.sqrt(1 - a)
    )

    df["trip_distance_km"] = EARTH_RADIUS_KM * c

    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering steps.
    """

    df = add_time_features(df)

    df = add_weekend_feature(df)

    df = add_peak_hour_feature(df)

    df = calculate_trip_distance(df)

    return df

