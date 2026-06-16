from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# TODO (S0-1) : chemin vers votre fichier de donnees (CSV) place dans data/
DATA_PATH = ROOT / "data" / "telco_churn_clean.csv"
MODEL_DIR = ROOT / "models"

# TODO (S0-2) : nom de la colonne cible binaire (valeurs 0/1)
TARGET = "Churn"

# TODO (S0-3) : colonnes numeriques de votre dataset
NUMERIC_FEATURES: list[str] = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

# TODO (S0-4) : colonnes categorielles (peut rester vide : [])
CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

RANDOM_STATE = 42

# Surcouche via variables d'environnement (principe 12-factor)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "telco-churn-baseline")
MODEL_NAME = os.getenv("MODEL_NAME", "telco-churn-classifier")

MLFLOW_EXPERIMENT_DESCRIPTION = os.getenv(
    "MLFLOW_EXPERIMENT_DESCRIPTION",
    "Projet de classification binaire pour predire le churn des clients telecoms.",
)

MLFLOW_EXPERIMENT_TAGS: dict[str, str] = {
    "project": "telco-customer-churn",
    "task": "binary-classification",
    "target": TARGET,
    "dataset": "Telco Customer Churn",
    "model_registry_name": MODEL_NAME,
}