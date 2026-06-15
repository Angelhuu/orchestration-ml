"""Frontend Streamlit (squelette) : tester l'API de classification.

Seance 14 bis - TP Streamlit
    Application minimale qui appelle l'API FastAPI (TP S12). Le formulaire de
    prediction est fourni partiellement : a vous d'adapter les champs a VOTRE
    dataset et de completer les TODO (S14bis-n).
    Lancement : `PYTHONPATH=todo streamlit run todo/frontend/app.py`

L'URL de l'API est lue depuis la variable d'environnement API_URL (utile en
docker compose, ou l'API est joignable via le nom de service `api`).
"""
from __future__ import annotations

import os

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Classification", layout="wide")
st.title("Demonstrateur de classification")

api_url = st.text_input("URL de l'API", value=API_URL)

predict_tab, history_tab = st.tabs(["Prediction", "Historique"])

with predict_tab:
    st.subheader("Tester l'endpoint /predict")

    with st.form("predict_form"):
        # TODO (S14bis-1) : remplacez ces champs d'exemple par les colonnes de
        #   VOTRE dataset (memes noms que le schema `Features` de l'API, cf TP
        #   S12). Utilisez st.number_input pour les variables numeriques et
        #   st.selectbox pour les categorielles.
        feature_numerique = st.number_input("feature_numerique", min_value=0.0, value=1.0)
        feature_categorielle = st.selectbox("feature_categorielle", ["A", "B", "C"])

        submitted = st.form_submit_button("Predire")

    if submitted:
        # TODO (S14bis-2) : construire le payload avec les memes cles que le
        #   schema `Features` de l'API (un dict {nom_colonne: valeur}).
        payload = {
            "feature_numerique": feature_numerique,
            "feature_categorielle": feature_categorielle,
        }
        try:
            response = httpx.post(f"{api_url}/predict", json=payload, timeout=10.0)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as exc:
            st.error(f"Appel a l'API impossible : {exc}")
        else:
            # TODO (S14bis-3) : afficher result["prediction"] (0/1) et
            #   result["probability"] (ex. st.metric, st.progress) au lieu du
            #   simple st.json ci-dessous.
            st.json(result)

with history_tab:
    st.subheader("Historique des previsions")
    # TODO (S14bis-4 bonus) : si vous ajoutez un endpoint GET /predictions a
    #   votre API (journal en base), recuperez-le ici avec httpx.get et
    #   affichez-le avec st.dataframe(pd.DataFrame(rows)). Sinon, laissez ce
    #   message.
    st.info("Aucun journal de previsions : ajoutez un endpoint /predictions a l'API (bonus).")
    _ = pd  # pandas est importe pour l'affichage du dataframe (bonus)
