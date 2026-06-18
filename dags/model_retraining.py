"""DAG Airflow : préparation des données, réentraînement et contrôle qualité."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

QUALITY_THRESHOLD = 0.55

PROJECT_ROOT = Path("/opt/airflow")
PREPARE_SCRIPT = PROJECT_ROOT / "scripts" / "prepare_telco.py"

default_args = {
    "owner": "HU Angel",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    logger.info("Préparation des données")
    subprocess.run(
        ["python", str(PREPARE_SCRIPT)],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def task_train(**context) -> None:
    from mlproject.train import train

    metrics = train(c=1.0, max_iter=1000)

    f1 = float(metrics["f1"])
    roc_auc = float(metrics["roc_auc"])

    logger.info("Modèle entraîné : f1=%.3f roc_auc=%.3f", f1, roc_auc)

    context["ti"].xcom_push(key="f1", value=f1)
    context["ti"].xcom_push(key="roc_auc", value=roc_auc)


def task_check_quality(**context) -> None:
    f1 = context["ti"].xcom_pull(task_ids="train", key="f1")

    if f1 is None:
        raise ValueError("F1-score introuvable dans XCom")

    f1 = float(f1)

    if f1 < QUALITY_THRESHOLD:
        raise ValueError(
            f"Qualité insuffisante : f1={f1:.3f} < seuil={QUALITY_THRESHOLD:.3f}"
        )

    logger.info("Contrôle qualité réussi : f1=%.3f", f1)


with DAG(
    dag_id="model_retraining",
    description="Prépare les données, réentraîne le modèle et contrôle sa qualité",
    schedule="0 3 * * 1",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["telco", "classification", "training"],
) as dag:
    prepare = PythonOperator(
        task_id="prepare_data",
        python_callable=task_prepare_data,
    )

    train_task = PythonOperator(
        task_id="train",
        python_callable=task_train,
    )

    check = PythonOperator(
        task_id="check_quality",
        python_callable=task_check_quality,
    )

    prepare >> train_task >> check
