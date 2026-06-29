

import streamlit as st
import pandas as pd

from src.config import (
    FEATURE_DATA_PATH,
    REPORT_DIR,
)


# ============================================================
# Data Loading
# ============================================================

@st.cache_data
def load_processed_data():
    """
    Load the processed feature-engineered dataset.

    Returns
    -------
    pd.DataFrame
    """

    return pd.read_csv(FEATURE_DATA_PATH)


@st.cache_data
def load_model_comparison():
    """
    Load model comparison report.

    Returns
    -------
    pd.DataFrame
    """

    return pd.read_csv(
        REPORT_DIR / "model_comparison.csv"
    )


@st.cache_data
def load_feature_importance():
    """
    Load feature importance report.

    Returns
    -------
    pd.DataFrame
    """

    return pd.read_csv(
        REPORT_DIR / "feature_importance.csv"
    )


@st.cache_data
def load_random_search_results():
    """
    Load Random Search CV results.

    Returns
    -------
    pd.DataFrame
    """

    return pd.read_csv(
        REPORT_DIR / "random_search_results.csv"
    )


# ============================================================
# Section Header
# ============================================================

def section_header(title, description=None):
    """
    Display a section title with an optional description.
    """

    st.header(title)

    if description:
        st.caption(description)

    st.divider()


# ============================================================
# KPI Card
# ============================================================

def metric_card(label, value, delta=None):
    """
    Display a metric card.
    """

    st.metric(
        label=label,
        value=value,
        delta=delta,
    )


# ============================================================
# Dataset Preview
# ============================================================

def dataset_preview(df, rows=15):
    """
    Display dataset preview.
    """

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(rows),
        use_container_width=True,
    )


# ============================================================
# Download Button
# ============================================================

def download_dataset(df):
    """
    Download dataset as CSV.
    """

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Dataset",
        data=csv,
        file_name="uber_dataset.csv",
        mime="text/csv",
    )