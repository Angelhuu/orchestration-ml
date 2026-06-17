"""Frontend Streamlit pour tester l'API Telco Churn."""

from __future__ import annotations

import os
from datetime import datetime

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Telco Churn Prediction", layout="wide")

st.title("Telco Churn Prediction")
st.write("Démonstrateur Streamlit pour prédire si un client risque de quitter l'opérateur.")

if "history" not in st.session_state:
    st.session_state["history"] = []

api_url = st.text_input("URL de l'API", value=API_URL)

predict_tab, info_tab, history_tab = st.tabs(
    ["Prediction", "Informations API", "Historique"]
)

with predict_tab:
    st.subheader("Tester l'endpoint /predict")

    with st.form("predict_form"):
        st.markdown("### Informations numériques")

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

        st.markdown("### Informations client")

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
            prediction = result["prediction"]
            probability = result["probability"]

            st.markdown("### Résultat")

            col1, col2 = st.columns(2)

            with col1:
                label = "Churn" if prediction == 1 else "No churn"
                st.metric("Classe prédite", label)

            with col2:
                st.metric("Probabilité de churn", f"{probability:.2%}")

            st.progress(probability)

            if prediction == 1:
                st.warning("Le client est prédit comme risquant de quitter l'opérateur.")
            else:
                st.success("Le client est prédit comme non-churn.")

            st.session_state["history"].append(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "prediction": prediction,
                    "probability": probability,
                    "Contract": Contract,
                    "InternetService": InternetService,
                    "tenure": tenure,
                    "MonthlyCharges": MonthlyCharges,
                    "TotalCharges": TotalCharges,
                }
            )

            with st.expander("Payload envoyé à l'API"):
                st.json(payload)

            with st.expander("Réponse brute de l'API"):
                st.json(result)

with info_tab:
    st.subheader("Informations sur l'API")

    if st.button("Tester /health"):
        try:
            response = httpx.get(f"{api_url}/health", timeout=5.0)
            response.raise_for_status()
            st.success("API disponible")
            st.json(response.json())
        except httpx.HTTPError as exc:
            st.error(f"API indisponible : {exc}")

    if st.button("Afficher /model-info"):
        try:
            response = httpx.get(f"{api_url}/model-info", timeout=5.0)
            response.raise_for_status()
            st.json(response.json())
        except httpx.HTTPError as exc:
            st.error(f"Impossible de récupérer les informations du modèle : {exc}")

    if st.button("Afficher /features"):
        try:
            response = httpx.get(f"{api_url}/features", timeout=5.0)
            response.raise_for_status()
            st.json(response.json())
        except httpx.HTTPError as exc:
            st.error(f"Impossible de récupérer les features : {exc}")

with history_tab:
    st.subheader("Historique local des prédictions")

    if st.session_state["history"]:
        history_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(history_df, use_container_width=True)

        if st.button("Vider l'historique"):
            st.session_state["history"] = []
            st.rerun()
    else:
        st.info("Aucune prédiction effectuée pour le moment.")
