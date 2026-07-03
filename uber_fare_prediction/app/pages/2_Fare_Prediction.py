"""Fare Prediction Page

Predict Uber fare using the trained Random Forest model.
"""

from __future__ import annotations

from pathlib import Path
import sys
from datetime import datetime, time

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.model import load_model
from src.predict import predict_fare
from src.dashboard_utils import footer, calculate_trip_distance, create_trip_summary
from src.validators import validate_coordinates


# ======================================================
# Load Model
# ======================================================

@st.cache_resource
def get_model():
    return load_model()


model = get_model()


# ======================================================
# Page Title
# ======================================================

st.title("🤖 Uber Fare Prediction")

st.markdown(
    """
Predict the estimated Uber fare by entering trip details below.
"""
)

st.divider()


# ======================================================
# Input Form
# ======================================================

with st.form("prediction_form"):
    st.subheader("Trip Information")

    passenger_count = st.number_input(
        "Passengers",
        min_value=1,
        max_value=6,
        value=1,
        step=1,
    )

    pickup_date = st.date_input(
        "Date",
        value=datetime.today(),
    )

    pickup_time = st.time_input(
        "Time",
        value=time(12, 0),
    )

    st.divider()

    st.subheader("Coordinates")

    col1, col2 = st.columns(2)
    with col1:
        pickup_latitude = st.number_input(
            "Pickup Latitude",
            value=40.7614327,
            format="%.6f",
        )
        dropoff_latitude = st.number_input(
            "Dropoff Latitude",
            value=40.6513111,
            format="%.6f",
        )

    with col2:
        pickup_longitude = st.number_input(
            "Pickup Longitude",
            value=-73.9798156,
            format="%.6f",
        )
        dropoff_longitude = st.number_input(
            "Dropoff Longitude",
            value=-73.8803331,
            format="%.6f",
        )

    sample_cols = st.columns([2, 1])
    with sample_cols[1]:
        if st.form_submit_button("✨ Sample Trip", use_container_width=True):
            pickup_latitude = 40.7614327
            pickup_longitude = -73.9798156
            dropoff_latitude = 40.6513111
            dropoff_longitude = -73.8803331
            passenger_count = 1
            pickup_date = datetime.today().date()
            pickup_time = time(12, 0)

    st.divider()

    submitted = st.form_submit_button("🚖 Predict Fare", use_container_width=True)


# ======================================================
# Trip Summary + Map
# ======================================================

pickup_datetime = datetime.combine(pickup_date, pickup_time)

# Validation
try:
    pickup_latitude, pickup_longitude = validate_coordinates(
        pickup_latitude, pickup_longitude
    )
    dropoff_latitude, dropoff_longitude = validate_coordinates(
        dropoff_latitude, dropoff_longitude
    )
except ValueError as e:
    st.error(str(e))
    st.stop()

# Distance + summary
trip_distance_km = calculate_trip_distance(
    pickup_latitude=pickup_latitude,
    pickup_longitude=pickup_longitude,
    dropoff_latitude=dropoff_latitude,
    dropoff_longitude=dropoff_longitude,
)

trip_summary = create_trip_summary(
    pickup_latitude=pickup_latitude,
    pickup_longitude=pickup_longitude,
    dropoff_latitude=dropoff_latitude,
    dropoff_longitude=dropoff_longitude,
    passenger_count=passenger_count,
    pickup_datetime=pickup_datetime,
)

st.subheader("Trip Summary")

sum_cols = st.columns(4)

sum_cols[0].metric("Pickup", trip_summary["Pickup"])
sum_cols[1].metric("Dropoff", trip_summary["Dropoff"])
sum_cols[2].metric("Distance", f"{trip_distance_km:.3f} km")
sum_cols[3].metric("Passengers", f"{passenger_count}")

st.caption(f"Pickup Time: {pickup_datetime}")

st.subheader("Pickup → Dropoff")

# Simple interactive map (st.map expects a dataframe/df-like with latitude/longitude columns)
import pandas as pd

data_points = pd.DataFrame(
    [
        {"latitude": pickup_latitude, "longitude": pickup_longitude, "label": "Pickup"},
        {"latitude": dropoff_latitude, "longitude": dropoff_longitude, "label": "Dropoff"},
    ]
)

st.map(data_points, latitude="latitude", longitude="longitude")



# ======================================================
# Prediction
# ======================================================

if submitted:
    input_data = {
        "key": "prediction",
        "pickup_datetime": pickup_datetime,
        "pickup_longitude": pickup_longitude,
        "pickup_latitude": pickup_latitude,
        "dropoff_longitude": dropoff_longitude,
        "dropoff_latitude": dropoff_latitude,
        "passenger_count": passenger_count,
    }

    with st.spinner("Predicting fare..."):
        prediction = predict_fare(model, input_data)

    st.divider()

    st.success(
        f"""
# 💰 Estimated Fare

## ${prediction:.2f}

**Trip Distance:** {trip_distance_km:.3f} km

**Estimated Model:** Random Forest
"""
    )

footer()

