"""DAG Airflow : envoie quotidien de prédictions à l'API."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

N_PREDICTIONS = 20

default_args = {
    "owner": "HU Angel",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_send_predictions(**context) -> None:
    import httpx

    from mlproject.config import API_URL, TARGET
    from mlproject.data import load_data

    df = load_data()
    features = df.drop(columns=[TARGET])

    n = min(N_PREDICTIONS, len(features))
    sample = features.sample(n=n, random_state=42)

    predictions = []

    with httpx.Client(base_url=API_URL, timeout=15.0) as client:
        client.get("/health").raise_for_status()

        for _, row in sample.iterrows():
            payload = json.loads(row.to_json())

            response = client.post("/predict", json=payload)
            response.raise_for_status()

            predictions.append(response.json())

    logger.info("%d prédictions envoyées à %s", len(predictions), API_URL)

    context["ti"].xcom_push(key="n_predictions", value=len(predictions))


with DAG(
    dag_id="daily_predictions",
    description="Envoie 20 prédictions par jour à l'API",
    default_args=default_args,
    start_date=datetime(2024, 1, 1),
    schedule="0 10 * * *",
    catchup=False,
    tags=["telco", "classification", "predictions"],
) as dag:
    send_predictions = PythonOperator(
        task_id="send_predictions",
        python_callable=task_send_predictions,
    )
