"""
tests/api/test_routes.py

API tests for FastAPI routes.
"""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


# ==========================================================
# Fixtures
# ==========================================================

VALID_REQUEST = {
    "pickup_longitude": -73.985428,
    "pickup_latitude": 40.748817,
    "dropoff_longitude": -73.985130,
    "dropoff_latitude": 40.758896,
    "passenger_count": 2,
    "pickup_datetime": "2015-05-02 08:30:00",
}


# ==========================================================
# Root Endpoint
# ==========================================================

def test_root():

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["application"] == "Uber Fare Prediction API"
    assert data["status"] == "running"
    assert data["version"] == "1.0.0"


# ==========================================================
# Health Endpoint
# ==========================================================

def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json() == {
        "status": "healthy",
    }


# ==========================================================
# Model Info Endpoint
# ==========================================================

def test_model_info():

    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "Random Forest Regressor"
    assert data["version"] == "1.0.0"
    assert data["features"] == 11


# ==========================================================
# Prediction Endpoint
# ==========================================================

def test_predict_success(monkeypatch):

    monkeypatch.setattr(
        "api.routes.make_prediction",
        lambda _: 18.75,
    )

    response = client.post(
        "/predict",
        json=VALID_REQUEST,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_fare"] == 18.75
    assert data["currency"] == "USD"
    assert data["model"] == "Random Forest Regressor"
    assert data["version"] == "1.0.0"


# ==========================================================
# Validation Tests
# ==========================================================

def test_invalid_pickup_latitude():

    payload = VALID_REQUEST.copy()

    payload["pickup_latitude"] = 100

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_dropoff_longitude():

    payload = VALID_REQUEST.copy()

    payload["dropoff_longitude"] = 200

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_invalid_passenger_count():

    payload = VALID_REQUEST.copy()

    payload["passenger_count"] = 8

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_missing_required_field():

    payload = VALID_REQUEST.copy()

    payload.pop("pickup_datetime")

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


# ==========================================================
# Exception Handling
# ==========================================================

def test_prediction_exception(monkeypatch):

    def fake_prediction(_):
        raise RuntimeError("Prediction failed")

    monkeypatch.setattr(
        "api.routes.make_prediction",
        fake_prediction,
    )

    response = client.post(
        "/predict",
        json=VALID_REQUEST,
    )

    assert response.status_code == 500

    assert response.json()["detail"] == "Prediction failed"