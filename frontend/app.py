"""Frontend Streamlit pour tester l'API Telco Churn."""

from __future__ import annotations

import os
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")
PUBLIC_API_URL = os.environ.get("PUBLIC_API_URL", "http://localhost:8000")
PUBLIC_MLFLOW_URL = os.environ.get("PUBLIC_MLFLOW_URL", "http://localhost:5000")


def clean_url(url: str) -> str:
    return url.rstrip("/")


def risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "Risque élevé"
    if probability >= 0.40:
        return "Risque moyen"
    return "Risque faible"


def build_risk_table(payload: dict, probability: float) -> pd.DataFrame:
    rows = []

    if payload["Contract"] == "Month-to-month":
        rows.append(
            {
                "Facteur": "Contract",
                "Valeur": payload["Contract"],
                "Interprétation": "Contrat court, souvent associé à plus de churn.",
            }
        )

    if payload["tenure"] <= 12:
        rows.append(
            {
                "Facteur": "tenure",
                "Valeur": payload["tenure"],
                "Interprétation": "Client récent, fidélité encore faible.",
            }
        )

    if payload["PaymentMethod"] == "Electronic check":
        rows.append(
            {
                "Facteur": "PaymentMethod",
                "Valeur": payload["PaymentMethod"],
                "Interprétation": "Mode de paiement souvent plus représenté chez les clients churn.",
            }
        )

    if payload["InternetService"] == "Fiber optic":
        rows.append(
            {
                "Facteur": "InternetService",
                "Valeur": payload["InternetService"],
                "Interprétation": "Peut être lié à des mensualités plus élevées.",
            }
        )

    if payload["MonthlyCharges"] >= 70:
        rows.append(
            {
                "Facteur": "MonthlyCharges",
                "Valeur": payload["MonthlyCharges"],
                "Interprétation": "Facture mensuelle élevée.",
            }
        )

    if payload["TechSupport"] == "No":
        rows.append(
            {
                "Facteur": "TechSupport",
                "Valeur": payload["TechSupport"],
                "Interprétation": "Absence de support technique.",
            }
        )

    if payload["OnlineSecurity"] == "No":
        rows.append(
            {
                "Facteur": "OnlineSecurity",
                "Valeur": payload["OnlineSecurity"],
                "Interprétation": "Absence de service de sécurité en ligne.",
            }
        )

    if not rows:
        rows.append(
            {
                "Facteur": "Profil global",
                "Valeur": "-",
                "Interprétation": "Aucun facteur de risque simple fortement visible.",
            }
        )

    rows.append(
        {
            "Facteur": "Probabilité finale",
            "Valeur": f"{probability:.2%}",
            "Interprétation": risk_level(probability),
        }
    )

    return pd.DataFrame(rows)


st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.title("Telco Churn Prediction")
st.write("Démonstrateur Streamlit pour prédire si un client risque de quitter l'opérateur.")

if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    st.header("Navigation")

    api_url = st.text_input("URL interne API", value=API_URL)
    public_api_url = st.text_input("URL publique API", value=PUBLIC_API_URL)
    public_mlflow_url = st.text_input("URL publique MLflow", value=PUBLIC_MLFLOW_URL)

    api_url = clean_url(api_url)
    public_api_url = clean_url(public_api_url)
    public_mlflow_url = clean_url(public_mlflow_url)

    st.markdown("### Accès rapides")
    st.markdown(f"- [Documentation API FastAPI]({public_api_url}/docs)")
    st.markdown(f"- [OpenAPI JSON]({public_api_url}/openapi.json)")
    st.markdown(f"- [MLflow UI]({public_mlflow_url})")

    st.markdown("### Endpoints")
    st.code("/health\n/predict\n/model-info\n/features")

predict_tab, info_tab, history_tab = st.tabs(
    ["Prédiction", "API & MLflow", "Historique"]
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
            MonthlyCharges = st.number_input(
                "MonthlyCharges",
                min_value=0.0,
                value=65.5,
                step=1.0,
            )

        with col4:
            TotalCharges = st.number_input(
                "TotalCharges",
                min_value=0.0,
                value=786.0,
                step=10.0,
            )

        st.markdown("### Variables catégorielles")

        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("gender", ["Female", "Male"])
            Partner = st.selectbox("Partner", ["Yes", "No"])
            Dependents = st.selectbox("Dependents", ["Yes", "No"])
            PhoneService = st.selectbox("PhoneService", ["Yes", "No"])
            MultipleLines = st.selectbox(
                "MultipleLines",
                ["Yes", "No", "No phone service"],
            )

        with col2:
            InternetService = st.selectbox(
                "InternetService",
                ["DSL", "Fiber optic", "No"],
            )
            OnlineSecurity = st.selectbox(
                "OnlineSecurity",
                ["Yes", "No", "No internet service"],
            )
            OnlineBackup = st.selectbox(
                "OnlineBackup",
                ["Yes", "No", "No internet service"],
            )
            DeviceProtection = st.selectbox(
                "DeviceProtection",
                ["Yes", "No", "No internet service"],
            )
            TechSupport = st.selectbox(
                "TechSupport",
                ["Yes", "No", "No internet service"],
            )

        with col3:
            StreamingTV = st.selectbox(
                "StreamingTV",
                ["Yes", "No", "No internet service"],
            )
            StreamingMovies = st.selectbox(
                "StreamingMovies",
                ["Yes", "No", "No internet service"],
            )
            Contract = st.selectbox(
                "Contract",
                ["Month-to-month", "One year", "Two year"],
            )
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
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()

        except httpx.HTTPError as exc:
            st.error(f"Appel à l'API impossible : {exc}")

        else:
            prediction = int(result["prediction"])
            probability = float(result["probability"])
            label = "Churn" if prediction == 1 else "No churn"
            level = risk_level(probability)

            st.markdown("## Résultat de la prédiction")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Classe prédite", label)

            with col2:
                st.metric("Probabilité de churn", f"{probability:.2%}")

            with col3:
                st.metric("Niveau de risque", level)

            st.progress(probability)

            if prediction == 1:
                st.warning("Le modèle prédit que le client risque de quitter l'opérateur.")
            else:
                st.success("Le modèle prédit que le client ne devrait pas quitter l'opérateur.")

            st.markdown("### Résumé tabulaire")

            summary_df = pd.DataFrame(
                [
                    {
                        "prediction": prediction,
                        "label": label,
                        "probability": probability,
                        "probability_percent": f"{probability:.2%}",
                        "risk_level": level,
                        "threshold": "0.50 par défaut",
                    }
                ]
            )
            st.dataframe(summary_df, use_container_width=True)

            st.markdown("### Interprétation simple des facteurs visibles")
            st.caption(
                "Ce tableau est une interprétation métier simple basée sur les valeurs du formulaire. "
                "Ce n'est pas une explication SHAP exacte du modèle."
            )

            risk_df = build_risk_table(payload, probability)
            st.dataframe(risk_df, use_container_width=True)

            st.markdown("### Données envoyées au modèle")

            input_df = pd.DataFrame(
                [{"feature": key, "value": value} for key, value in payload.items()]
            )
            st.dataframe(input_df, use_container_width=True)

            st.session_state["history"].append(
                {
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
                }
            )

            with st.expander("Réponse brute de l'API"):
                st.json(result)

            with st.expander("Payload JSON envoyé"):
                st.json(payload)

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
            try:
                response = httpx.get(f"{api_url}/health", timeout=5.0)
                response.raise_for_status()
                st.success("API disponible")
                st.json(response.json())
            except httpx.HTTPError as exc:
                st.error(f"API indisponible : {exc}")

    with col2:
        if st.button("Afficher /model-info"):
            try:
                response = httpx.get(f"{api_url}/model-info", timeout=5.0)
                response.raise_for_status()
                st.json(response.json())
            except httpx.HTTPError as exc:
                st.error(f"Impossible de récupérer /model-info : {exc}")

    with col3:
        if st.button("Afficher /features"):
            try:
                response = httpx.get(f"{api_url}/features", timeout=5.0)
                response.raise_for_status()
                st.json(response.json())
            except httpx.HTTPError as exc:
                st.error(f"Impossible de récupérer /features : {exc}")

with history_tab:
    st.subheader("Historique local des prédictions")

    if st.session_state["history"]:
        history_df = pd.DataFrame(st.session_state["history"])

        st.markdown("### Tableau des prédictions")
        st.dataframe(history_df, use_container_width=True)

        st.markdown("### Statistiques rapides")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Nombre de prédictions", len(history_df))

        with col2:
            st.metric("Probabilité moyenne", f"{history_df['probability'].mean():.2%}")

        with col3:
            churn_rate = (history_df["prediction"].mean()) if len(history_df) else 0
            st.metric("Taux de churn prédit", f"{churn_rate:.2%}")

        st.markdown("### Répartition des classes prédites")
        class_counts = history_df["label"].value_counts()
        st.bar_chart(class_counts)

        csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Télécharger l'historique CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv",
        )

        if st.button("Vider l'historique"):
            st.session_state["history"] = []
            st.rerun()

    else:
        st.info("Aucune prédiction effectuée pour le moment.")
