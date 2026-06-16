"""API d'inference d'un modele de classification FastAPI."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from mlproject.config import MODEL_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_path = MODEL_DIR / "model.joblib"

    if not model_path.exists():
        logger.error("Modele introuvable : %s", model_path)
        raise FileNotFoundError(
            f"Modele introuvable : {model_path}. Lancez d'abord make train ou make train-models."
        )

    logger.info("Chargement du modele depuis %s", model_path)
    ml["model"] = joblib.load(model_path)

    yield

    logger.info("Nettoyage du modele charge")
    ml.clear()


app = FastAPI(
    title="Telco Churn Classification API",
    version="0.1.0",
    lifespan=lifespan,
)


class Features(BaseModel):
    SeniorCitizen: int = Field(..., ge=0, le=1, description="1 si le client est senior, sinon 0")
    tenure: int = Field(..., ge=0, description="Anciennete du client en mois")
    MonthlyCharges: float = Field(..., ge=0, description="Montant mensuel facture")
    TotalCharges: float = Field(..., ge=0, description="Montant total facture")

    gender: Literal["Female", "Male"]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "SeniorCitizen": 0,
                    "tenure": 12,
                    "MonthlyCharges": 65.5,
                    "TotalCharges": 786.0,
                    "gender": "Female",
                    "Partner": "Yes",
                    "Dependents": "No",
                    "PhoneService": "Yes",
                    "MultipleLines": "No",
                    "InternetService": "Fiber optic",
                    "OnlineSecurity": "No",
                    "OnlineBackup": "Yes",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "StreamingTV": "Yes",
                    "StreamingMovies": "Yes",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "PaymentMethod": "Electronic check",
                }
            ]
        }
    }


class PredictionOut(BaseModel):
    prediction: int = Field(..., description="Classe predite : 0 = pas de churn, 1 = churn")
    probability: float = Field(..., description="Probabilite estimee de churn")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = ml.get("model")

    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    row = pd.DataFrame([features.model_dump()])

    try:
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as exc:
        logger.exception("Erreur pendant la prediction")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PredictionOut(
        prediction=int(proba >= 0.5),
        probability=round(proba, 4),
    )


@app.get("/model-info")
def model_info() -> dict:
    return {
        "version": os.environ.get("MODEL_VERSION", "unknown"),
        "model_path": str(MODEL_DIR / "model.joblib"),
    }