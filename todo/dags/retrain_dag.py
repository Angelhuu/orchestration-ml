"""DAG Airflow - pipeline de re-entrainement du modele (squelette).

Seance 17 - TP Airflow
    Pipeline simple : preparation des donnees -> entrainement -> controle
    qualite. Completez les TODO (S17-n).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

logger = logging.getLogger(__name__)

# f1 minimal du modele entraine pour que le pipeline soit considere comme reussi.
QUALITY_THRESHOLD = 0.65

default_args = {
    "owner": "data-team",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_prepare_data(**context) -> None:
    # TODO (S17-1) : appeler le script/fonction qui (re)genere ou prepare votre
    #               jeu de donnees dans data/ (ex. votre script de preparation)
    raise NotImplementedError


def task_train(**context) -> None:
    # TODO (S17-2) :
    #   - importer mlproject.train.train et l'appeler -> metrics = train()
    #   - pousser metrics["f1"] dans XCom : context["ti"].xcom_push(key="f1", value=...)
    raise NotImplementedError


def task_check_quality(**context) -> None:
    # TODO (S17-3) :
    #   - recuperer f1 = context["ti"].xcom_pull(task_ids="train", key="f1")
    #   - si f1 < QUALITY_THRESHOLD, lever une ValueError (le pipeline echoue)
    #   - sinon, logger un message de succes
    raise NotImplementedError


with DAG(
    dag_id="model_retraining",
    description="Prepare les donnees, reentraine le modele et controle sa qualite",
    # TODO (S17-4) : definir le planning, ex. schedule="0 3 * * 1" (tous les lundis a 3h)
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["classification", "training"],
) as dag:
    prepare = PythonOperator(task_id="prepare_data", python_callable=task_prepare_data)
    train_task = PythonOperator(task_id="train", python_callable=task_train)
    check = PythonOperator(task_id="check_quality", python_callable=task_check_quality)

    # TODO (S17-5) : declarer l'ordre d'execution : prepare >> train_task >> check
