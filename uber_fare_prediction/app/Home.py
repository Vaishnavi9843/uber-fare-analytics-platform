"""streamlit_app.py

Main entry point for the Uber Fare Analytics Platform.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.styles import load_css
from src.dashboard_utils import get_dataset_statistics

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Uber Fare Analytics Platform",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()

# --------------------------------------------------
# Sidebar (single block)
# --------------------------------------------------

with st.sidebar:
    st.title("🚖 Uber Analytics")
    st.caption("Machine Learning • Data Engineering • MLOps")
    st.divider()

    st.markdown("### Project Workflow")
    st.markdown(
        """
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

stats = get_dataset_statistics()

col1, col2 = st.columns(2)

with col1:
    st.metric("Rows", f"{stats['rows']:,}")
    st.metric("Total Features", f"{stats['features']}")

with col2:
    st.metric("Average Fare", f"${stats['avg_fare']:.2f}")
    st.metric("Average Distance", f"{stats['avg_distance']:.2f} km")

st.metric("Average Passengers", f"{stats['avg_passengers']:.2f}")

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

```
"""
)

st.divider()

# --------------------------------------------------
# Project Status
# --------------------------------------------------

st.header("📌 Project Status")

st.markdown(
    """
🟢 Data Pipeline

🟢 Machine Learning

🟢 Dashboard

🟡 FastAPI (Upcoming)

🟡 Docker

🟡 PostgreSQL

🟡 MLflow
"""
)

