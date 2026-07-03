"""validators.py

Lightweight validation helpers.

These are intentionally small and focused so they can be reused both in the
Streamlit app and later in a FastAPI service.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def validate_required_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")


def validate_latitude(latitude: float) -> float:
    """Validate latitude for coordinates-based inputs."""

    try:
        lat = float(latitude)
    except (TypeError, ValueError) as e:
        raise ValueError("Latitude must be a number") from e

    if not (-90.0 <= lat <= 90.0):
        raise ValueError("Latitude must be between -90 and 90")
    return lat


def validate_longitude(longitude: float) -> float:
    """Validate longitude for coordinates-based inputs."""

    try:
        lon = float(longitude)
    except (TypeError, ValueError) as e:
        raise ValueError("Longitude must be a number") from e

    if not (-180.0 <= lon <= 180.0):
        raise ValueError("Longitude must be between -180 and 180")
    return lon


def validate_coordinates(latitude: float, longitude: float) -> tuple[float, float]:
    """Validate a pair of (latitude, longitude) and return normalized floats."""

    lat = validate_latitude(latitude)
    lon = validate_longitude(longitude)
    return lat, lon






