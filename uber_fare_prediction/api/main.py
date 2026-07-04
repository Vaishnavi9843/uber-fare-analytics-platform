"""
Main FastAPI application.
"""

from fastapi import FastAPI

from api.routes import router

app = FastAPI(
    title="Uber Fare Prediction API",
    description="REST API for predicting Uber fares using a trained Random Forest model.",
    version="1.0.0",
)

app.include_router(router)