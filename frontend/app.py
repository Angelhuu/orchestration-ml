from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
PUBLIC_API_URL = os.environ.get("PUBLIC_API_URL", "http://localhost:8000")
PUBLIC_MLFLOW_URL = os.environ.get("PUBLIC_MLFLOW_URL", "http://localhost:5000")
PUBLIC_AIRFLOW_URL = os.environ.get("PUBLIC_AIRFLOW_URL", "http://localhost:8080")
EVAL_DATA_PATH = Path(os.environ.get("EVAL_DATA_PATH", "/app/data/telco_churn_clean.csv"))

TARGET = "Churn"

FEATURE_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
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


def clean_url(url: str) -> str:
    return url.rstrip("/")


def risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "Risque élevé"
    if probability >= 0.40:
        return "Risque moyen"
    return "Risque faible"


def to_jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return value


def build_payload(row: pd.Series) -> dict[str, Any]:
    return {col: to_jsonable(row[col]) for col in FEATURE_COLUMNS}


def call_predict(api_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = httpx.post(f"{api_url}/predict", json=payload, timeout=15.0)
    response.raise_for_status()
    return response.json()


def build_risk_table(payload: dict[str, Any], probability: float) -> pd.DataFrame:
    rows = []

    if payload["Contract"] == "Month-to-month":
        rows.append({
            "Facteur": "Contract",
            "Valeur": payload["Contract"],
            "Interprétation": "Contrat mensuel : risque de churn souvent plus élevé.",
        })

    if payload["tenure"] <= 12:
        rows.append({
            "Facteur": "tenure",
            "Valeur": payload["tenure"],
            "Interprétation": "Client récent : fidélité encore faible.",
        })

    if payload["PaymentMethod"] == "Electronic check":
        rows.append({
            "Facteur": "PaymentMethod",
            "Valeur": payload["PaymentMethod"],
            "Interprétation": "Mode de paiement souvent associé à plus de churn.",
        })

    if payload["MonthlyCharges"] >= 70:
        rows.append({
            "Facteur": "MonthlyCharges",
            "Valeur": payload["MonthlyCharges"],
            "Interprétation": "Facture mensuelle élevée.",
        })

    if payload["TechSupport"] == "No":
        rows.append({
            "Facteur": "TechSupport",
            "Valeur": payload["TechSupport"],
            "Interprétation": "Absence de support technique.",
        })

    if not rows:
        rows.append({
            "Facteur": "Profil global",
            "Valeur": "-",
            "Interprétation": "Aucun facteur de risque simple fortement visible.",
        })

    rows.append({
        "Facteur": "Probabilité finale",
        "Valeur": f"{probability:.2%}",
        "Interprétation": risk_level(probability),
    })

    return pd.DataFrame(rows)


st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.title("Telco Churn Prediction")
st.caption("Projet réalisé par HU Angel")
st.write("Démonstrateur Streamlit pour prédire si un client risque de quitter l'opérateur.")

if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    st.header("Navigation")

    api_url = clean_url(st.text_input("URL interne API", value=API_URL))
    public_api_url = clean_url(st.text_input("URL publique API", value=PUBLIC_API_URL))
    public_mlflow_url = clean_url(st.text_input("URL publique MLflow", value=PUBLIC_MLFLOW_URL))
    public_airflow_url = clean_url(st.text_input("URL publique Airflow", value=PUBLIC_AIRFLOW_URL))

    st.markdown("### Accès rapides")
    st.markdown(f"- [Documentation API FastAPI]({public_api_url}/docs)")
    st.markdown(f"- [OpenAPI JSON]({public_api_url}/openapi.json)")
    st.markdown(f"- [MLflow UI]({public_mlflow_url})")
    st.markdown(f"- [Airflow UI]({public_airflow_url})")

predict_tab, eval_tab, info_tab, airflow_tab, history_tab = st.tabs(
    ["Prédiction", "Évaluation", "API & MLflow", "Airflow", "Historique"]
)

with predict_tab:
    st.subheader("Tester l'endpoint /predict")

    with st.form("predict_form"):
        st.markdown("### Variables numériques")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            SeniorCitizen = st.selectbox("SeniorCitizen", [0, 1], index=0)
        with col2:
            tenure = st.number_input("tenure", min_value=0, max_value=100, value=12)
        with col3:
            MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, value=65.5, step=1.0)
        with col4:
            TotalCharges = st.number_input("TotalCharges", min_value=0.0, value=786.0, step=10.0)

        st.markdown("### Variables catégorielles")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("gender", ["Female", "Male"])
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
            MultipleLines = st.selectbox("MultipleLines", ["Yes", "No", "No phone service"])

        with col2:
            InternetService = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
            OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
            OnlineBackup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
            DeviceProtection = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])

        with col3:
            StreamingTV = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])
            Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes", "No"])
            PaymentMethod = st.selectbox(
                "PaymentMethod",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )

        submitted = st.form_submit_button("Prédire")

    if submitted:
        payload = {
            "SeniorCitizen": SeniorCitizen,
            "tenure": tenure,
            "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges,
            "gender": gender,
            "Partner": Partner,
            "Dependents": Dependents,
            "PhoneService": PhoneService,
            "MultipleLines": MultipleLines,
            "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity,
            "OnlineBackup": OnlineBackup,
            "DeviceProtection": DeviceProtection,
            "TechSupport": TechSupport,
            "StreamingTV": StreamingTV,
            "StreamingMovies": StreamingMovies,
            "Contract": Contract,
            "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod,
        }

        try:
            result = call_predict(api_url, payload)
        except httpx.HTTPError as exc:
            st.error(f"Appel à l'API impossible : {exc}")
        else:
            prediction = int(result["prediction"])
            probability = float(result["probability"])
            label = "Churn" if prediction == 1 else "No churn"
            level = risk_level(probability)

            st.markdown("## Résultat de la prédiction")

            col1, col2, col3 = st.columns(3)
            col1.metric("Classe prédite", label)
            col2.metric("Probabilité de churn", f"{probability:.2%}")
            col3.metric("Niveau de risque", level)

            st.progress(probability)

            if prediction == 1:
                st.warning("Le modèle prédit que le client risque de quitter l'opérateur.")
            else:
                st.success("Le modèle prédit que le client ne devrait pas quitter l'opérateur.")

            st.markdown("### Interprétation simple")
            st.dataframe(build_risk_table(payload, probability), use_container_width=True)

            st.markdown("### Données envoyées au modèle")
            input_df = pd.DataFrame([{"feature": k, "value": v} for k, v in payload.items()])
            st.dataframe(input_df, use_container_width=True)

            st.session_state["history"].append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "prediction": prediction,
                "label": label,
                "probability": probability,
                "risk_level": level,
                "Contract": Contract,
                "InternetService": InternetService,
                "PaymentMethod": PaymentMethod,
                "tenure": tenure,
                "MonthlyCharges": MonthlyCharges,
                "TotalCharges": TotalCharges,
            })

with eval_tab:
    st.subheader("Évaluation du modèle")

    st.write("Cette page évalue l'API sur un échantillon du dataset propre.")

    if not EVAL_DATA_PATH.exists():
        st.error(f"Dataset d'évaluation introuvable : {EVAL_DATA_PATH}")
    else:
        max_samples = st.slider(
            "Nombre de lignes à évaluer",
            min_value=50,
            max_value=1000,
            value=200,
            step=50,
        )

        if st.button("Lancer l'évaluation"):
            df = pd.read_csv(EVAL_DATA_PATH)

            eval_df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)

            y_true = eval_df[TARGET].astype(int).tolist()
            y_pred = []
            y_proba = []

            progress = st.progress(0.0)

            for i, row in eval_df.iterrows():
                payload = build_payload(row)
                result = call_predict(api_url, payload)

                y_pred.append(int(result["prediction"]))
                y_proba.append(float(result["probability"]))

                progress.progress((i + 1) / len(eval_df))

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)

            try:
                roc_auc = roc_auc_score(y_true, y_proba)
            except ValueError:
                roc_auc = None

            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{accuracy:.3f}")
            col2.metric("F1-score", f"{f1:.3f}")
            col3.metric("ROC-AUC", "N/A" if roc_auc is None else f"{roc_auc:.3f}")

            st.markdown("### Matrice de confusion")

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(
                cm,
                index=["Vrai No churn", "Vrai Churn"],
                columns=["Prédit No churn", "Prédit Churn"],
            )
            st.dataframe(cm_df, use_container_width=True)

            st.markdown("### Classification report")

            report = classification_report(
                y_true,
                y_pred,
                target_names=["No churn", "Churn"],
                output_dict=True,
                zero_division=0,
            )
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

            st.markdown("### Exemples de prédictions")

            examples = eval_df[FEATURE_COLUMNS].copy()
            examples["true_label"] = y_true
            examples["predicted_label"] = y_pred
            examples["churn_probability"] = y_proba
            st.dataframe(examples.head(20), use_container_width=True)

with info_tab:
    st.subheader("Accès API et MLflow")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### FastAPI")
        st.markdown(f"[Ouvrir Swagger /docs]({public_api_url}/docs)")
        st.markdown(f"[Ouvrir OpenAPI JSON]({public_api_url}/openapi.json)")

    with col2:
        st.markdown("### MLflow")
        st.markdown(f"[Ouvrir MLflow UI]({public_mlflow_url})")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Tester /health"):
            response = httpx.get(f"{api_url}/health", timeout=5.0)
            st.json(response.json())

    with col2:
        if st.button("Afficher /model-info"):
            response = httpx.get(f"{api_url}/model-info", timeout=5.0)
            st.json(response.json())

    with col3:
        if st.button("Afficher /features"):
            response = httpx.get(f"{api_url}/features", timeout=5.0)
            st.json(response.json())

with airflow_tab:
    st.subheader("Orchestration avec Airflow")

    st.write(
        "Airflow permet de planifier et surveiller les pipelines du projet : "
        "réentraînement du modèle et génération de prédictions quotidiennes."
    )

    st.markdown("### Accès Airflow")
    st.markdown(f"[Ouvrir Airflow UI]({public_airflow_url})")

    st.info("Identifiants : username = admin, password = admin")

    st.markdown("### DAGs disponibles")

    dags_df = pd.DataFrame(
        [
            {
                "DAG": "model_retraining",
                "Rôle": "Prépare les données, réentraîne le modèle et vérifie le F1-score.",
                "Planning": "Tous les lundis à 03h00",
            },
            {
                "DAG": "daily_predictions",
                "Rôle": "Envoie un lot de prédictions à l'API FastAPI.",
                "Planning": "Tous les jours à 10h00",
            },
        ]
    )

    st.dataframe(dags_df, use_container_width=True)

    st.markdown("### Commandes utiles")

    st.code(
        "docker compose -f docker-compose.yml -f docker-compose.airflow.yml ps\n"
        "docker compose -f docker-compose.yml -f docker-compose.airflow.yml logs --tail=100 airflow-webserver\n"
        "docker compose -f docker-compose.yml -f docker-compose.airflow.yml logs --tail=100 airflow-scheduler",
        language="bash",
    )


with history_tab:
    st.subheader("Historique local des prédictions")

    if st.session_state["history"]:
        history_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(history_df, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Nombre de prédictions", len(history_df))
        col2.metric("Probabilité moyenne", f"{history_df['probability'].mean():.2%}")
        col3.metric("Taux de churn prédit", f"{history_df['prediction'].mean():.2%}")

        st.bar_chart(history_df["label"].value_counts())

        if st.button("Vider l'historique"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("Aucune prédiction effectuée pour le moment.")
