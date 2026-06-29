"""
Fare Prediction Page

Predict Uber fare using the trained Random Forest model.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from datetime import datetime, time

from src.model import load_model
from src.predict import predict_fare

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
        "Passenger Count",
        min_value=1,
        max_value=6,
        value=1,
    )

    pickup_date = st.date_input(
        "Pickup Date",
        value=datetime.today(),
    )

    pickup_time = st.time_input(
        "Pickup Time",
        value=time(12, 0),
    )

    st.divider()

    st.subheader("Pickup Location")

    col1, col2 = st.columns(2)

    with col1:

        pickup_latitude = st.number_input(
            "Pickup Latitude",
            value=40.7614327,
            format="%.6f",
        )

    with col2:

        pickup_longitude = st.number_input(
            "Pickup Longitude",
            value=-73.9798156,
            format="%.6f",
        )

    st.divider()

    st.subheader("Dropoff Location")

    col3, col4 = st.columns(2)

    with col3:

        dropoff_latitude = st.number_input(
            "Dropoff Latitude",
            value=40.6513111,
            format="%.6f",
        )

    with col4:

        dropoff_longitude = st.number_input(
            "Dropoff Longitude",
            value=-73.8803331,
            format="%.6f",
        )

    submitted = st.form_submit_button(
        "🚖 Predict Fare"
    )

# ======================================================
# Prediction
# ======================================================

if submitted:

    pickup_datetime = datetime.combine(
        pickup_date,
        pickup_time,
    )

    input_data = {

        "key": "prediction",

        "pickup_datetime": pickup_datetime,

        "pickup_longitude": pickup_longitude,

        "pickup_latitude": pickup_latitude,

        "dropoff_longitude": dropoff_longitude,

        "dropoff_latitude": dropoff_latitude,

        "passenger_count": passenger_count,

    }

    prediction = predict_fare(
        model,
        input_data,
    )

    st.divider()

    st.success(
        f"Estimated Uber Fare: **${prediction:.2f}**"
    )