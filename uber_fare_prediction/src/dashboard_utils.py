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
# Section Header
# ============================================================

def section_header(title: str, description: str | None = None) -> None:
    """Display a section title with an optional description."""

    st.header(title)

    if description:
        st.caption(description)

    st.divider()


# ============================================================
# KPI Card
# ============================================================

def metric_card(label: str, value, delta=None) -> None:
    """Display a metric card."""

    st.metric(label=label, value=value, delta=delta)


# ============================================================
# Dataset Preview
# ============================================================

def dataset_preview(df: pd.DataFrame, rows: int = 15) -> None:
    """Display dataset preview."""

    st.subheader("Dataset Preview")

    st.dataframe(df.head(rows), use_container_width=True)


# ============================================================
# Download Button
# ============================================================

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

