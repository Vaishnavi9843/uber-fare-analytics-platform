"""
API routes.
"""

from fastapi import APIRouter, HTTPException

from api.predictor import make_prediction
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)

router = APIRouter()


@router.get(
    "/",
)
def root():

    return {
        "application": "Uber Fare Prediction API",
        "version": "1.0.0",
        "status": "running",
    }


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health():

    return HealthResponse(
        status="healthy",
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
)
def model_info():

    return ModelInfoResponse(
        model="Random Forest Regressor",
        version="1.0.0",
        features=11,
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(request: PredictionRequest):

    try:

        prediction = make_prediction(
            request.model_dump(),
        )

        return PredictionResponse(
            predicted_fare=prediction,
            currency="USD",
            model="Random Forest Regressor",
            version="1.0.0",
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )