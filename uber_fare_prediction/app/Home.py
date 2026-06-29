"""
streamlit_app.py

Main entry point for the Uber Fare Analytics Platform.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import streamlit as st


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Uber Fare Analytics Platform",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🚖 Uber Fare Analytics Platform")

st.sidebar.success(
    "Select a page from the sidebar."
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
### Project Workflow

- Data Understanding
- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Model Training
- Model Optimization
- Interactive Dashboard
"""
)


# --------------------------------------------------
# Main Title
# --------------------------------------------------

st.title("🚖 Uber Fare Analytics Platform")

st.markdown(
"""
### End-to-End Machine Learning & Data Analytics Project

This platform predicts Uber fares using Machine Learning while also providing interactive analytics and insights into historical trip data.

The application was built as a complete ML workflow—from raw data preprocessing to model deployment using Streamlit.
"""
)


st.divider()


# --------------------------------------------------
# Project Overview
# --------------------------------------------------

st.header("📌 Project Overview")

st.write(
"""
The project follows an end-to-end machine learning pipeline.

It includes:

- Data Understanding
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Hyperparameter Optimization
- Model Evaluation
- Interactive Dashboard
"""
)


# --------------------------------------------------
# Dataset Information
# --------------------------------------------------

st.header("📊 Dataset")

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Rows",
        "199,999"
    )

    st.metric(
        "Features",
        "11"
    )

with col2:

    st.metric(
        "Target Variable",
        "Fare Amount"
    )

    st.metric(
        "Model",
        "Random Forest"
    )


st.info(
"""
The dataset contains Uber trip records including:

- Pickup Coordinates
- Dropoff Coordinates
- Pickup Date & Time
- Passenger Count
- Fare Amount

Additional temporal and spatial features were engineered to improve model performance.
"""
)

st.divider()


# --------------------------------------------------
# Technology Stack
# --------------------------------------------------

st.header("🛠 Technology Stack")

col1, col2, col3 = st.columns(3)

with col1:

    st.subheader("Data Processing")

    st.markdown(
        """
- Python
- Pandas
- NumPy
"""
    )

with col2:

    st.subheader("Machine Learning")

    st.markdown(
        """
- Scikit-Learn
- Random Forest
- Joblib
"""
    )

with col3:

    st.subheader("Visualization")

    st.markdown(
        """
- Streamlit
- Plotly
"""
    )

st.divider()


# --------------------------------------------------
# Project Pipeline
# --------------------------------------------------

st.header("⚙ Project Pipeline")

st.markdown(
"""
```text
Uber Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Random Forest Model
      │
      ▼
Streamlit Dashboard

"""
)

st.divider()
