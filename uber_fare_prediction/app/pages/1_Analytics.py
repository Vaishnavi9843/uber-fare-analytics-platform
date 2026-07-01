"""
Analytics Dashboard

Interactive dashboard for exploring the Uber dataset.
"""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import plotly.express as px

from src.dashboard_utils import (
    load_processed_data,
    metric_card,
    section_header,
    dataset_preview,
    download_dataset,
    footer,
)

# ======================================================
# Load Dataset
# ======================================================

df = load_processed_data()

# ======================================================
# Page Config
# ======================================================

st.title("📊 Analytics Dashboard")

st.markdown(
    """
Explore the processed Uber trip dataset using interactive visualizations
and summary statistics.
"""
)

st.divider()

# ======================================================
# KPI Cards
# ======================================================

section_header(
    "Dataset Overview",
    "Summary statistics of the processed dataset."
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card(
        "Total Trips",
        f"{len(df):,}"
    )

with col2:
    metric_card(
        "Average Fare",
        f"${df['fare_amount'].mean():.2f}"
    )

with col3:
    metric_card(
        "Average Distance",
        f"{df['trip_distance_km'].mean():.2f} km"
    )

with col4:
    metric_card(
        "Average Passengers",
        f"{df['passenger_count'].mean():.2f}"
    )

st.divider()

# ======================================================
# Fare Distribution
# ======================================================

section_header(
    "Fare Distribution"
)

fig = px.histogram(
    df,
    x="fare_amount",
    nbins=50,
    title="Distribution of Uber Fare Amount",
    template="plotly_white",
)

fig.update_layout(
    title_x=0.5
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# Trip Distance Distribution
# ======================================================

section_header(
    "Trip Distance Distribution"
)

fig = px.histogram(
    df,
    x="trip_distance_km",
    nbins=50,
    title="Distribution of Trip Distance",
    template="plotly_white",
)

fig.update_layout(
    title_x=0.5
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# Trips by Hour
# ======================================================

section_header(
    "Trips by Hour"
)

hour_df = (
    df.groupby("pickup_hour")
    .size()
    .reset_index(name="Trips")
)

fig = px.bar(
    hour_df,
    x="pickup_hour",
    y="Trips",
    title="Trips by Pickup Hour",
    template="plotly_white",
)

fig.update_layout(
    title_x=0.5
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# Trips by Day
# ======================================================

section_header(
    "Trips by Day"
)

day_order = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

day_df = (
    df.groupby("pickup_day")
    .size()
    .reindex(day_order)
    .reset_index(name="Trips")
)

fig = px.bar(
    day_df,
    x="pickup_day",
    y="Trips",
    title="Trips by Day",
    template="plotly_white",
)

fig.update_layout(
    title_x=0.5
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# Correlation Heatmap
# ======================================================

section_header(
    "Correlation Heatmap"
)

numeric_df = df.select_dtypes(include="number")

corr = numeric_df.corr()

fig = px.imshow(
    corr,
    text_auto=".2f",
    color_continuous_scale="RdBu_r",
    aspect="auto",
)

fig.update_layout(
    title="Feature Correlation",
    title_x=0.5,
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ======================================================
# Dataset Preview
# ======================================================

section_header(
    "Dataset Preview"
)

dataset_preview(df)

download_dataset(df)

footer()