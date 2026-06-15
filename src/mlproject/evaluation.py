"""Outils d'evaluation partages : graphiques loggues comme artefacts MLflow."""
from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def log_shap_summary(pipeline: Pipeline, x_test, name: str, max_samples: int = 200) -> None:
    """Logger un summary plot SHAP comme artefact MLflow ``shap_summary.png``.

    Parameters
    ----------
    pipeline : Pipeline
        Pipeline entraine, avec les etapes ``preprocessor`` et ``clf``.
    x_test : pandas.DataFrame
        Jeu de test utilise pour estimer les valeurs SHAP.
    name : str
        Nom du modele, utilise dans le titre du graphique.
    max_samples : int, optional
        Nombre maximal d'observations utilisees pour le calcul, par defaut 200
        (limite le temps de calcul sur les modeles a base d'arbres).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    clf = pipeline.named_steps["clf"]

    transformed = preprocessor.transform(x_test)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    feature_names = preprocessor.get_feature_names_out()
    sample = transformed[:max_samples]

    try:
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(sample)
    except Exception:  # pragma: no cover - modeles non supportes par TreeExplainer
        logger.warning("SHAP TreeExplainer indisponible pour %s, artefact ignore", name)
        return

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    shap.summary_plot(shap_values, sample, feature_names=feature_names, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Importance des variables (SHAP) : {name}")
    mlflow.log_figure(fig, "shap_summary.png")
    plt.close(fig)
