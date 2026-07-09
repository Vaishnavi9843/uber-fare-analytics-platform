<img width="1024" height="1536" alt="Uber_architecture" src="https://github.com/user-attachments/assets/cc8900c0-94ff-4d21-9499-097014969689" />
# 🚖 Uber Fare Analytics Platform

> End-to-End Machine Learning, Data Analytics & MLOps Project

Predict Uber fares using Machine Learning while exploring historical trip data through an interactive dashboard built with Streamlit, FastAPI, Docker, and GitHub Actions.

> **Replace the image placeholders with your screenshots stored under `docs/images/`.**

---

# Badges

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Docker](https://img.shields.io/badge/Docker-Container-blue)
![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI-success)
![Pytest](https://img.shields.io/badge/Pytest-Testing-yellow)
![MIT](https://img.shields.io/badge/License-MIT-purple)

---

# Table of Contents

1. Project Overview
2. Features
3. Screenshots
4. Architecture
5. ML Workflow
6. Tech Stack
7. Project Structure
8. Installation
9. Running the Project
10. Docker
11. API Documentation
12. Testing
13. CI/CD
14. Performance
15. Dataset
16. Future Improvements
17. License
18. Author

---

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

# Screenshot Placeholders

Create:

docs/images/

and add:

- home.png
- analytics.png
- prediction.png
- performance.png
- about.png
- swagger.png
- docker-build.png
- github-actions.png
- architecture.png

Then replace these:

![Home](docs/images/home.png)

![Analytics](docs/images/analytics.png)

![Prediction](docs/images/prediction.png)

![Performance](docs/images/performance.png)

![About](docs/images/about.png)

![Swagger](docs/images/swagger.png)

![Docker](docs/images/docker-build.png)

![GitHub Actions](docs/images/github-actions.png)

---

# Architecture

![Architecture](docs/images/architecture.png)

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
