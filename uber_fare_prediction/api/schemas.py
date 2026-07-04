"""
Pydantic schemas for request and response models.
"""

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    pickup_longitude: float = Field(..., ge=-180, le=180)
    pickup_latitude: float = Field(..., ge=-90, le=90)

    dropoff_longitude: float = Field(..., ge=-180, le=180)
    dropoff_latitude: float = Field(..., ge=-90, le=90)

    passenger_count: int = Field(..., ge=1, le=6)

    pickup_datetime: str


class PredictionResponse(BaseModel):
    predicted_fare: float


class HealthResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    model: str
    version: str
    features: int