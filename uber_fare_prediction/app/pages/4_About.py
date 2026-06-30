"""
About Page

Project documentation and technical overview.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.title("📖 About This Project")

st.markdown(
"""
An end-to-end Machine Learning application for Uber fare prediction built
using modern Data Science, Software Engineering, and MLOps principles.

The project demonstrates the complete lifecycle of an ML system—from raw
data analysis to a deployable prediction dashboard.
"""
)

st.divider()

# ==========================================================
# Project Overview
# ==========================================================

st.header("🎯 Project Overview")

st.markdown("""
The objective of this project is to estimate Uber ride fares based on trip
details such as pickup location, dropoff location, passenger count,
and pickup time.

Instead of building only a machine learning model, the project focuses on
developing a complete production-ready pipeline that includes:

- Data preprocessing
- Feature engineering
- Model training
- Hyperparameter optimization
- Interactive dashboard
- Modular project architecture

This mirrors how machine learning systems are developed in industry.
""")

# ==========================================================
# Business Problem
# ==========================================================

st.header("💼 Business Problem")

st.markdown("""
Ride-sharing companies generate millions of trips every day.

Accurately estimating fares is important for:

- Customer price transparency
- Route planning
- Driver earnings estimation
- Demand forecasting
- Business analytics

This project predicts trip fares using historical ride information and
engineered spatial and temporal features.
""")

# ==========================================================
# Solution Architecture
# ==========================================================

st.header("🏗️ Solution Architecture")

st.code(
"""
Raw Uber Dataset
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
Prediction Pipeline
        │
        ▼
Interactive Streamlit Dashboard
""",
language="text"
)

# ==========================================================
# Machine Learning Pipeline
# ==========================================================

st.header("🧠 Machine Learning Pipeline")

st.markdown("""
The project follows a structured ML workflow:

1. Data Understanding
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Engineering
5. Model Training
6. Hyperparameter Optimization
7. Model Evaluation
8. Model Serialization
9. Interactive Deployment
""")

# ==========================================================
# Technology Stack
# ==========================================================

st.header("🛠️ Technology Stack")

col1, col2 = st.columns(2)

with col1:

    st.markdown("""
### Data Science

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib
""")

with col2:

    st.markdown("""
### Visualization & App

- Plotly
- Streamlit
- Git
- GitHub
""")

# ==========================================================
# Project Structure
# ==========================================================

st.header("📂 Project Structure")

st.code("""
uber_fare_prediction/
│
├── app/
│   ├── Home.py
│   └── pages/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── predict.py
│   ├── model.py
│   ├── dashboard_utils.py
│
├── data/
├── models/
├── notebooks/
├── reports/
└── requirements.txt
""", language="text")

# ==========================================================
# Key Features
# ==========================================================

st.header("✨ Key Features")

st.markdown("""
- Interactive analytics dashboard
- Real-time fare prediction
- Modular machine learning pipeline
- Feature engineering automation
- Hyperparameter optimization
- Model evaluation dashboard
- Reusable project architecture
""")

# ==========================================================
# Challenges
# ==========================================================

st.header("⚡ Challenges & Solutions")

st.markdown("""
| Challenge | Solution |
|-----------|----------|
| Missing values | Data cleaning pipeline |
| Invalid coordinates | Coordinate validation |
| Feature creation | Automated engineering functions |
| Model optimization | RandomizedSearchCV |
| Code organization | Modular `src/` package |
| Deployment readiness | Streamlit application |
""")

# ==========================================================
# Future Roadmap
# ==========================================================

st.header("🚀 Future Roadmap")

st.markdown("""
The current version focuses on Machine Learning and application development.

The next phases include:

- Docker containerization
- FastAPI prediction service
- PostgreSQL integration
- Airflow ETL pipelines
- MLflow experiment tracking
- CI/CD automation
- Cloud deployment
- Monitoring and logging
""")

# ==========================================================
# Developer Notes
# ==========================================================

st.header("👨‍💻 Developer Notes")

st.info("""
This project was intentionally developed in multiple stages to mirror a
real-world machine learning lifecycle. Each phase was completed,
validated, and version-controlled before progressing to the next stage.

The result is a maintainable, modular, and extensible machine learning
application that serves as a foundation for future Data Engineering and
MLOps enhancements.
""")