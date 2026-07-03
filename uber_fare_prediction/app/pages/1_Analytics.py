"""Analytics Dashboard

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
    download_dataframe,
    filter_dataset,
    footer,
)


df = load_processed_data()

st.title("📊 Analytics Dashboard")

st.markdown(
    """
Explore the processed Uber trip dataset using interactive visualizations
and summary statistics.
"""
)

st.divider()

with st.sidebar:
    st.header("Filters")

    passenger_min, passenger_max = st.slider(
        "Passenger Count",
        min_value=1,
        max_value=6,
        value=(1, 6),
        step=1,
    )

    fare_min = float(df["fare_amount"].min())
    fare_max = float(df["fare_amount"].max())
    fare_lo, fare_hi = st.slider(
        "Fare Range",
        min_value=float(fare_min),
        max_value=float(fare_max),
        value=(float(fare_min), float(fare_max)),
    )

    dist_min = float(df["trip_distance_km"].min())
    dist_max = float(df["trip_distance_km"].max())
    dist_lo, dist_hi = st.slider(
        "Distance Range (km)",
        min_value=float(dist_min),
        max_value=float(dist_max),
        value=(float(dist_min), float(dist_max)),
    )

    hour_min, hour_max = int(df["pickup_hour"].min()), int(df["pickup_hour"].max())
    hour_lo, hour_hi = st.slider(
        "Hour Range",
        min_value=hour_min,
        max_value=hour_max,
        value=(hour_min, hour_max),
        step=1,
    )

    weekend_only = st.checkbox("Weekend Only", value=False)


filtered_df = filter_dataset(
    df,
    fare_range=(fare_lo, fare_hi),
    distance_range=(dist_lo, dist_hi),
    passenger_count=(passenger_min, passenger_max),
    hour_range=(hour_lo, hour_hi),
    weekend_only=weekend_only,
)

# KPI Cards
section_header(
    "Dataset Overview",
    "Summary statistics of the filtered dataset.",
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    metric_card("Total Trips", f"{len(filtered_df):,}")

with col2:
    metric_card("Average Fare", f"${filtered_df['fare_amount'].mean():.2f}")

with col3:
    metric_card(
        "Average Distance",
        f"{filtered_df['trip_distance_km'].mean():.2f} km",
    )

with col4:
    metric_card(
        "Average Passengers",
        f"{filtered_df['passenger_count'].mean():.2f}",
    )

st.divider()

if len(filtered_df) == 0:
    st.warning("No trips match the selected filters.")
    st.stop()

# Tabs

tabs = st.tabs(
    [
        "Overview",
        "Temporal Analysis",
        "Spatial Analysis",
        "Correlation",
        "Dataset",
    ]
)

# Overview
with tabs[0]:
    section_header("Overview - Distributions")

    # Histogram: keep existing
    fig = px.histogram(
        filtered_df,
        x="fare_amount",
        nbins=50,
        title="Distribution of Uber Fare Amount",
        template="plotly_white",
    )
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig, width="stretch")

    section_header("Trip Distance Distribution")
    fig = px.histogram(
        filtered_df,
        x="trip_distance_km",
        nbins=50,
        title="Distribution of Trip Distance",
        template="plotly_white",
    )
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig,width="stretch")

    section_header("Fare vs Distance")
    fig = px.scatter(
        filtered_df,
        x="trip_distance_km",
        y="fare_amount",
        opacity=0.7,
        title="Fare vs Distance",
        template="plotly_white",
    )
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig, width="stretch")

    section_header("Fare by Passenger Count")
    fig = px.box(
        filtered_df,
        x="passenger_count",
        y="fare_amount",
        points="outliers",
        title="Fare by Passenger Count",
        template="plotly_white",
    )
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig, width="stretch")

# Temporal
with tabs[1]:
    section_header("Trips by Hour")
    hour_df = (
        filtered_df.groupby("pickup_hour").size().reset_index(name="Trips")
    )
    fig = px.bar(
        hour_df,
        x="pickup_hour",
        y="Trips",
        title="Trips by Pickup Hour",
        template="plotly_white",
    )
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig, width="stretch")

    section_header("Average Fare by Hour")
    avg_hour_df = (
        filtered_df.groupby("pickup_hour")["fare_amount"].mean().reset_index(name="Avg Fare")
    )
    fig = px.line(
        avg_hour_df,
        x="pickup_hour",
        y="Avg Fare",
        markers=True,
        title="Average Fare by Hour",
        template="plotly_white",
    )
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig,width="stretch")

    section_header("Trips by Day")
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
        filtered_df.groupby("pickup_day")
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
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig, width="stretch")

    section_header("Weekend vs Weekday Trips")
    if "is_weekend" in filtered_df.columns:
        weekend_counts = filtered_df["is_weekend"].value_counts().to_dict()
        weekend = weekend_counts.get(1, 0)
        weekday = weekend_counts.get(0, 0)
    else:
        # fallback
        weekend_days = {"Saturday", "Sunday"}
        weekend = filtered_df[filtered_df["pickup_day"].isin(weekend_days)].shape[0]
        weekday = filtered_df.shape[0] - weekend

    pie_df = px.data.tips()
    pie_df = None

    pie_values = [weekday, weekend]
    pie_labels = ["Weekday", "Weekend"]

    fig = px.pie(
        names=pie_labels,
        values=pie_values,
        title="Weekend vs Weekday Trips",
        template="plotly_white",
    )
    fig.update_layout(title_x=0.5)
    st.plotly_chart(fig, width="stretch")

# Spatial
with tabs[2]:
    section_header("Spatial Analysis")

    # Existing heatmap (kept): using pickup lat/long as a proxy heatmap
    # NOTE: requires engineered columns; we fall back gracefully.
    lat_col = "pickup_latitude" if "pickup_latitude" in filtered_df.columns else None
    lon_col = "pickup_longitude" if "pickup_longitude" in filtered_df.columns else None

    if lat_col and lon_col:
        section_header("Heatmap")
        fig = px.density_mapbox(
            filtered_df,
            lat=lat_col,
            lon=lon_col,
            z="fare_amount" if "fare_amount" in filtered_df.columns else None,
            radius=10,
            center=dict(lat=float(filtered_df[lat_col].mean()), lon=float(filtered_df[lon_col].mean())),
            zoom=11,
            mapbox_style="open-street-map",
            title="Pickup Density (Colored by Fare)",
        )
        fig.update_layout(title_x=0.5)
        st.plotly_chart(fig,width="stretch")
    else:
        st.info("Spatial heatmap columns not found in the processed dataset.")

# Correlation
with tabs[3]:
    section_header("Correlation Heatmap")
    numeric_df = filtered_df.select_dtypes(include="number")
    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
    )
    fig.update_layout(title="Feature Correlation", title_x=0.5)
    st.plotly_chart(fig, width="stretch")

# Dataset
with tabs[4]:
    section_header("Dataset Preview")
    dataset_preview(filtered_df)

    section_header("Download")
    csv_file = "uber_filtered_dataset.csv"
    download_dataframe(filtered_df, filename=csv_file)

    footer()

