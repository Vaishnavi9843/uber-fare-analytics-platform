<img width="1024" height="1536" alt="Uber_architecture" src="https://github.com/user-attachments/assets/6b3f0abb-00d0-4e3b-9f7b-01da6d41971d" />


# Project Overview

This repository demonstrates a production-style end-to-end Machine Learning project for Uber fare prediction.

It includes:

- Data preprocessing
- Exploratory data analysis
- Feature engineering
- Random Forest model training
- Hyperparameter optimization
- Model evaluation
- Interactive Streamlit dashboard
- FastAPI REST API
- Docker containerization
- Automated testing with Pytest
- Continuous Integration using GitHub Actions

---

# Features

- Interactive Analytics Dashboard
- Real-time Fare Prediction
- FastAPI Backend
- Swagger Documentation
- Docker Compose Deployment
- Automated CI Pipeline
- Unit, API and Integration Tests
- Modular Project Structure

---

## Layers

Data Layer
- Raw dataset
- Cleaning
- Feature Engineering
- Processed dataset

Machine Learning Layer
- Train/Test split
- Random Forest
- Hyperparameter tuning
- Evaluation

Backend
- FastAPI
- Validation
- Prediction
- JSON response

Frontend
- Streamlit dashboard

Deployment
- Docker Compose
- FastAPI container
- Streamlit container

CI/CD
- GitHub Actions
- Automated testing

---

# ML Workflow

Raw Data
→ Cleaning
→ EDA
→ Feature Engineering
→ Training
→ Hyperparameter Optimization
→ Evaluation
→ Saved Model
→ FastAPI
→ Streamlit
→ Docker
→ GitHub Actions

---

# Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- FastAPI
- Streamlit
- Docker
- Docker Compose
- Pytest
- GitHub Actions

---

# Project Structure

```text
uber_fare_prediction/
├── api/
├── app/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── tests/
├── Dockerfile.fastapi
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

# Installation

```bash
git clone https://github.com/Vaishnavi9843/uber-fare-analytics-platform.git

cd uber-fare-analytics-platform/uber_fare_prediction

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

---

# Run Locally

FastAPI

```bash
uvicorn api.main:app --reload
```

Streamlit

```bash
streamlit run app/Home.py
```

---

# Docker

```bash
docker compose up --build
```

Stop

```bash
docker compose down
```

---

# API

GET /
GET /health
GET /model-info
POST /predict

Swagger:

http://localhost:8000/docs

---

# Testing

```bash
pytest
```

Coverage

```bash
pytest --cov
```

Includes:
- Unit Tests
- API Tests
- Integration Tests

---

# CI/CD

GitHub Actions automatically:

- Installs dependencies
- Runs tests
- Validates every push

Workflow:

.github/workflows/ci.yml

---

# Model Performance

| Metric | Value |
|---|---:|
| MAE | 1.778 |
| RMSE | 3.708 |
| R² | 0.852 |

---

# Dataset

Historical Uber trips including:

- Passenger Count
- Pickup Location
- Dropoff Location
- Date
- Time
- Distance
- Fare Amount

---

# Future Improvements

- Cloud Deployment
- Kubernetes
- MLflow
- Drift Detection
- Authentication
- Real-time Predictions

---

# License

MIT License

---

# Author

**Vaishnavi Gaikwad**

GitHub:
https://github.com/Vaishnavi9843

If you like this project, consider giving it a ⭐.
