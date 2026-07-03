from __future__ import annotations

import streamlit as st
import pandas as pd

from src.config import FEATURE_DATA_PATH, REPORT_DIR
from src.logger import logger


# ============================================================
# Data Loading
# ============================================================

@st.cache_data
def load_processed_data() -> pd.DataFrame:
    """Load the processed feature-engineered dataset."""

    try:
        return pd.read_csv(FEATURE_DATA_PATH)
    except FileNotFoundError:
        logger.error("Processed dataset not found: %s", FEATURE_DATA_PATH)
        st.error(
            "Processed dataset not found. Please generate it using the notebook pipeline."
        )
        st.stop()


@st.cache_data
def load_model_comparison() -> pd.DataFrame:
    """Load model comparison report."""

    path = REPORT_DIR / "model_comparison.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Report not found: %s", path)
        st.error("Model comparison report not found.")
        st.stop()


@st.cache_data
def load_feature_importance() -> pd.DataFrame:
    """Load feature importance report."""

    path = REPORT_DIR / "feature_importance.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Report not found: %s", path)
        st.error("Feature importance report not found.")
        st.stop()


@st.cache_data
def load_random_search_results() -> pd.DataFrame:
    """Load Random Search CV results."""

    path = REPORT_DIR / "random_search_results.csv"
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        logger.error("Report not found: %s", path)
        st.error("Random search results not found.")
        st.stop()


# ============================================================
# Dataset helpers (stats / filtering / summaries)
# ============================================================


def calculate_trip_distance(
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
) -> float:
    """Calculate great-circle distance (km) between pickup and dropoff."""

    # Haversine formula
    import math

    r = 6371.0

    phi1 = math.radians(float(pickup_latitude))
    phi2 = math.radians(float(dropoff_latitude))

    d_phi = math.radians(float(dropoff_latitude) - float(pickup_latitude))
    d_lambda = math.radians(float(dropoff_longitude) - float(pickup_longitude))

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return round(r * c, 3)


def get_dataset_statistics() -> dict[str, float]:
    """Return basic summary statistics for the processed dataset."""

    df = load_processed_data()

    rows = len(df)
    features = int(df.shape[1])

    avg_fare = float(df["fare_amount"].mean())
    avg_distance = float(df["trip_distance_km"].mean())
    avg_passengers = float(df["passenger_count"].mean())

    return {
        "rows": rows,
        "features": features,
        "avg_fare": avg_fare,
        "avg_distance": avg_distance,
        "avg_passengers": avg_passengers,
    }


def filter_dataset(
    df: pd.DataFrame,
    fare_range: tuple[float, float] | None = None,
    distance_range: tuple[float, float] | None = None,
    passenger_count: tuple[int, int] | None = None,
    hour_range: tuple[int, int] | None = None,
    weekend_only: bool = False,
) -> pd.DataFrame:
    """Filter dataset based on common analytics controls."""

    filtered = df.copy()

    if fare_range is not None:
        lo, hi = fare_range
        filtered = filtered[
            (filtered["fare_amount"] >= lo) & (filtered["fare_amount"] <= hi)
        ]

    if distance_range is not None:
        lo, hi = distance_range
        filtered = filtered[
            (filtered["trip_distance_km"] >= lo)
            & (filtered["trip_distance_km"] <= hi)
        ]

    if passenger_count is not None:
        lo, hi = passenger_count
        filtered = filtered[
            (filtered["passenger_count"] >= lo)
            & (filtered["passenger_count"] <= hi)
        ]

    if hour_range is not None:
        lo, hi = hour_range
        filtered = filtered[
            (filtered["pickup_hour"] >= lo) & (filtered["pickup_hour"] <= hi)
        ]

    if weekend_only:
        if "is_weekend" in filtered.columns:
            filtered = filtered[filtered["is_weekend"] == 1]
        elif "pickup_day" in filtered.columns:
            weekend_days = {"Saturday", "Sunday"}
            filtered = filtered[filtered["pickup_day"].isin(weekend_days)]

    return filtered


def create_trip_summary(
    *,
    pickup_latitude: float,
    pickup_longitude: float,
    dropoff_latitude: float,
    dropoff_longitude: float,
    passenger_count: int | float,
    pickup_datetime: object | None = None,
) -> dict[str, object]:
    """Create a reusable trip summary for UI and API."""

    distance_km = calculate_trip_distance(
        pickup_latitude=pickup_latitude,
        pickup_longitude=pickup_longitude,
        dropoff_latitude=dropoff_latitude,
        dropoff_longitude=dropoff_longitude,
    )

    summary: dict[str, object] = {
        "Distance": distance_km,
        "Pickup": f"({pickup_latitude}, {pickup_longitude})",
        "Dropoff": f"({dropoff_latitude}, {dropoff_longitude})",
        "Passenger Count": passenger_count,
    }

    if pickup_datetime is not None:
        summary["Pickup Time"] = pickup_datetime

    return summary


# ============================================================
# UI components
# ============================================================


def section_header(title: str, description: str | None = None) -> None:
    """Display a section title with an optional description."""

    st.header(title)

    if description:
        st.caption(description)

    st.divider()


def metric_card(label: str, value, delta=None) -> None:
    """Display a metric card."""

    st.metric(label=label, value=value, delta=delta)


def dataset_preview(df: pd.DataFrame, rows: int = 15) -> None:
    """Display dataset preview."""

    st.subheader("Dataset Preview")

    st.dataframe(df.head(rows), use_container_width=True)


def download_dataframe(df: pd.DataFrame, filename: str) -> None:
    """Download a dataframe as CSV."""

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Dataset",
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


# Backward-compatible wrapper

def download_dataset(df: pd.DataFrame) -> None:
    """Download the default Uber dataset as CSV."""

    download_dataframe(df, filename="uber_dataset.csv")


def footer() -> None:
    st.divider()

    st.caption("Built with Python • Streamlit • Scikit-Learn • Plotly")

