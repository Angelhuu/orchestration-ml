"""API d'inference d'un modele de classification FastAPI."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from mlproject.config import CATEGORICAL_FEATURES, MODEL_DIR, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = MODEL_DIR / "model.joblib"
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

try:
    CHURN_THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.5"))
except ValueError:
    logger.warning("CHURN_THRESHOLD invalide, utilisation de 0.5")
    CHURN_THRESHOLD = 0.5

ml: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charger le modele une seule fois au demarrage de l'API."""
    if not MODEL_PATH.exists():
        logger.error("Modele introuvable : %s", MODEL_PATH)
        raise FileNotFoundError(
            f"Modele introuvable : {MODEL_PATH}. "
            "Lancez d'abord make train ou make train-models."
        )

    logger.info("Chargement du modele depuis %s", MODEL_PATH)
    ml["model"] = joblib.load(MODEL_PATH)

    yield

    logger.info("Nettoyage du modele charge")
    ml.clear()


app = FastAPI(
    title="Telco Churn Classification API",
    description="API FastAPI pour predire le churn des clients telecoms.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8501",
        "http://localhost:8501",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def get_model() -> Any:
    model = ml.get("model")

    if model is None:
        raise HTTPException(status_code=503, detail="Modele non charge")

    return model


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model_loaded": ml.get("model") is not None,
    }


@app.post("/predict", response_model=PredictionOut)
def predict(features: Features) -> PredictionOut:
    model = get_model()

    payload = features.model_dump()
    row = pd.DataFrame([payload])

    # Garantit que les colonnes envoyees au modele sont exactement celles de l'entrainement.
    row = row[FEATURE_COLUMNS]

    try:
        proba = float(model.predict_proba(row)[0, 1])
    except Exception as exc:
        logger.exception("Erreur pendant la prediction")
        raise HTTPException(
            status_code=500,
            detail="Erreur interne pendant la prediction",
        ) from exc

    return PredictionOut(
        prediction=int(proba >= CHURN_THRESHOLD),
        probability=round(proba, 4),
    )


@app.get("/model-info")
def model_info() -> dict:
    return {
        "version": os.environ.get("MODEL_VERSION", "unknown"),
        "model_path": str(MODEL_PATH),
        "model_loaded": ml.get("model") is not None,
        "threshold": CHURN_THRESHOLD,
        "features": FEATURE_COLUMNS,
        "n_features": len(FEATURE_COLUMNS),
    }


@app.get("/features")
def features_info() -> dict:
    return {
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "all_features": FEATURE_COLUMNS,
    }