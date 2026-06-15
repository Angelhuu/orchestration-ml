"""API d'inference d'un modele de classification (FastAPI).

Seance 12 - TP FastAPI
    /health est fourni et fonctionne. A vous d'implementer le schema d'entree
    (adapte a VOTRE jeu de donnees), le schema de sortie, le chargement du
    modele et l'endpoint /predict (voir les TODO S12-n).
    Lancement : `uvicorn mlproject.api:app --reload`
"""
from __future__ import annotations

import logging
import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# TODO (S12-3) : importer joblib, asynccontextmanager (contextlib),
#               AsyncIterator (typing) et MODEL_DIR (mlproject.config)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ml: dict = {}


# TODO (S12-3) : definir lifespan(app), decore par @asynccontextmanager :
#   - au demarrage : ml["model"] = joblib.load(MODEL_DIR / "model.joblib")
#   - yield
#   - a l'arret : ml.clear()
# Cela permet de charger le modele une seule fois (pas a chaque requete).


app = FastAPI(title="Classification API", version="0.1.0")  # TODO (S12-3) : ajouter lifespan=lifespan


class Features(BaseModel):
    # TODO (S12-1) : remplacez ces champs d'exemple par les colonnes de VOTRE
    #   dataset (memes noms que NUMERIC_FEATURES + CATEGORICAL_FEATURES dans
    #   config.py), avec leurs types et contraintes (ex. Field(..., ge=0)).
    #
    #   Ajoutez aussi un exemple de payload pour enrichir la doc /docs :
    #   model_config = {
    #       "json_schema_extra": {
    #           "examples": [{"feature_numerique": 1.0, "feature_categorielle": "A"}]
    #       }
    #   }
    feature_numerique: float = Field(..., description="Exemple de variable numerique")
    feature_categorielle: str = Field(..., description="Exemple de variable categorielle")


# TODO (S12-2) : definir PredictionOut(BaseModel) avec deux champs :
#   - prediction : int   (classe predite, 0 ou 1)
#   - probability : float


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# TODO (S12-4) : definir l'endpoint de prediction :
#
# @app.post("/predict", response_model=PredictionOut)
# def predict(features: Features) -> PredictionOut:
#     model = ml.get("model")
#     if model is None:
#         raise HTTPException(status_code=503, detail="Modele non charge")
#     row = pd.DataFrame([features.model_dump()])
#     proba = float(model.predict_proba(row)[0, 1])
#     return PredictionOut(prediction=int(proba >= 0.5), probability=round(proba, 4))


# TODO (S12-5 bonus) : definir GET /model-info retournant la version servie,
#               lue depuis la variable d'environnement MODEL_VERSION :
#
# @app.get("/model-info")
# def model_info() -> dict:
#     return {"version": os.environ.get("MODEL_VERSION", "unknown")}
