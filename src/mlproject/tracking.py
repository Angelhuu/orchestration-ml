from __future__ import annotations

import logging

import mlflow
import mlflow.data
import pandas as pd

from mlproject.config import (
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MLFLOW_EXPERIMENT_DESCRIPTION,
    MLFLOW_EXPERIMENT_TAGS,
    MLFLOW_TRACKING_URI,
    TARGET,
)

logger = logging.getLogger(__name__)


def setup_experiment() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    experiment = mlflow.set_experiment(MLFLOW_EXPERIMENT)
    client = mlflow.MlflowClient()

    if MLFLOW_EXPERIMENT_DESCRIPTION:
        client.set_experiment_tag(
            experiment.experiment_id,
            "mlflow.note.content",
            MLFLOW_EXPERIMENT_DESCRIPTION,
        )

    for key, value in MLFLOW_EXPERIMENT_TAGS.items():
        client.set_experiment_tag(
            experiment.experiment_id,
            key,
            str(value),
        )

    logger.info("Experience MLflow configuree : %s", MLFLOW_EXPERIMENT)


def log_dataset(df: pd.DataFrame, context: str, name: str = "dataset") -> None:
    dataset = mlflow.data.from_pandas(
        df,
        source=str(DATA_PATH),
        targets=TARGET,
        name=name,
    )

    mlflow.log_input(dataset, context=context)