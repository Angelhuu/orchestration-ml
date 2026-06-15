"""Optimisation d'hyperparametres avec Optuna.

Seance 6 - TP Optuna
    Ce module optimise les hyperparametres de trois familles de modeles
    (Random Forest, XGBoost, LightGBM) avec Optuna (sampler TPE), compare
    leurs performances et persiste le meilleur dans `models/model.joblib`.
    Completez les TODO (S6-n).

Chaque famille est suivie dans MLflow (un run par famille, imbrique sous un
run parent) et la meilleure est enregistree dans le Model Registry.

Lancement :
    python -m mlproject.train_optuna
    python -m mlproject.train_optuna --n-trials 50 --cv 3
    python -m mlproject.train_optuna --no-mlflow   # desactive le suivi MLflow
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# TODO (S6-1) : importer optuna, optuna.samplers et
#               sklearn.model_selection.cross_val_score

from mlproject.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_NAME,
)
from mlproject.data import load_data, split
from mlproject.evaluation import log_shap_summary
from mlproject.features import build_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    """Specification d'une famille de modeles a optimiser avec Optuna.

    Parameters
    ----------
    name : str
        Identifiant lisible de la famille de modeles.
    suggest_params : Callable[[optuna.Trial], dict]
        Construit un jeu d'hyperparametres pour un essai donne.
    build_estimator : Callable[[dict], ClassifierMixin]
        Construit l'estimateur scikit-learn a partir d'un jeu d'hyperparametres.
    """

    name: str
    suggest_params: Callable
    build_estimator: Callable[[dict], ClassifierMixin]


def build_model_specs() -> list[ModelSpec]:
    """Construire la liste des familles de modeles a optimiser.

    Returns
    -------
    list of ModelSpec
        Random Forest, XGBoost et LightGBM avec leurs espaces de recherche.
    """
    # TODO (S6-2) : importer RandomForestClassifier (sklearn.ensemble), XGBClassifier
    #               (xgboost), LGBMClassifier (lightgbm) et RANDOM_STATE
    #               (mlproject.config), puis pour chaque famille definir suggest_params
    #               (espace de recherche via trial.suggest_*) et build_estimator
    #               (construction de l'estimateur a partir des hyperparametres,
    #               avec cast(ClassifierMixin, ...) si necessaire) :
    #   - random_forest : n_estimators (int, ex. 100-300), max_depth
    #     (categorical, ex. [None, 10, 20, 30]), min_samples_leaf (int, ex. 1-5)
    #   - xgboost       : n_estimators (int, ex. 100-300), max_depth (int, ex. 3-10),
    #     learning_rate (float, echelle log, ex. 0.01-0.3)
    #   - lightgbm      : n_estimators (int, ex. 50-300), num_leaves (int, ex. 15-127),
    #     learning_rate (float, echelle log, ex. 0.01-0.3), max_depth (int, ex. 3-12)
    raise NotImplementedError


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    """Assembler le preprocessing et un classifieur dans un pipeline.

    Parameters
    ----------
    estimator : ClassifierMixin
        Classifieur place en derniere etape (``clf``).

    Returns
    -------
    Pipeline
        Pipeline scikit-learn pret a etre optimise.
    """
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", estimator),
        ]
    )


def objective(trial, spec: ModelSpec, x_train, y_train, cv: int) -> float:
    """Fonction objectif Optuna : score moyen de validation croisee.

    Parameters
    ----------
    trial : optuna.Trial
        Essai courant.
    spec : ModelSpec
        Famille de modeles optimisee.
    x_train, y_train : array-like
        Donnees d'entrainement.
    cv : int
        Nombre de plis de validation croisee.

    Returns
    -------
    float
        Score ROC AUC moyen sur les plis de validation croisee (a maximiser).
    """
    # TODO (S6-3) : appeler spec.suggest_params(trial) puis spec.build_estimator(params),
    #               construire le pipeline avec build_pipeline, evaluer avec
    #               cross_val_score (scoring="roc_auc", cv=cv) et retourner la moyenne
    raise NotImplementedError


def run_study(spec: ModelSpec, x_train, y_train, n_trials: int, cv: int):
    """Lancer l'etude Optuna pour une famille de modeles.

    Parameters
    ----------
    spec : ModelSpec
        Famille de modeles a optimiser.
    x_train, y_train : array-like
        Donnees d'entrainement.
    n_trials : int
        Nombre d'essais a evaluer.
    cv : int
        Nombre de plis de validation croisee passe a `objective`.

    Returns
    -------
    optuna.Study
        Etude Optuna une fois l'optimisation terminee.
    """
    # TODO (S6-4) : creer l'etude avec optuna.create_study
    #               (direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    # TODO (S6-5) : lancer study.optimize sur `n_trials` essais, en appelant
    #               objective(trial, spec, x_train, y_train, cv) pour chaque essai
    raise NotImplementedError


@dataclass
class FamilyResult:
    """Resultat d'optimisation d'une famille de modeles.

    Parameters
    ----------
    spec : ModelSpec
        Famille de modeles optimisee.
    study : optuna.Study
        Etude Optuna terminee.
    best_pipeline : Pipeline
        Pipeline reentraine avec les meilleurs hyperparametres.
    test_roc_auc : float
        ROC AUC sur le jeu de test.
    preds : np.ndarray
        Predictions (classes) sur le jeu de test.
    """

    spec: ModelSpec
    study: Any
    best_pipeline: Pipeline
    test_roc_auc: float
    preds: np.ndarray


def optimize_family(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
) -> FamilyResult:
    """Optimiser une famille de modeles avec Optuna et l'evaluer sur le test.

    Parameters
    ----------
    spec : ModelSpec
        Famille de modeles a optimiser.
    x_train, y_train : array-like
        Donnees d'entrainement.
    x_test, y_test : array-like
        Donnees de test pour l'evaluation finale.
    n_trials : int
        Nombre d'essais Optuna.
    cv : int
        Nombre de plis de validation croisee.

    Returns
    -------
    FamilyResult
        Meilleur pipeline et metriques associees.
    """
    logger.info("Optimisation de %s (n_trials=%d, cv=%d)", spec.name, n_trials, cv)
    study = run_study(spec, x_train, y_train, n_trials=n_trials, cv=cv)

    best_pipeline = build_pipeline(spec.build_estimator(study.best_params))
    best_pipeline.fit(x_train, y_train)
    proba = best_pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    test_roc_auc = float(roc_auc_score(y_test, proba))

    logger.info(
        "%s : cv_roc_auc=%.3f  test_roc_auc=%.3f  params=%s",
        spec.name,
        study.best_value,
        test_roc_auc,
        study.best_params,
    )
    return FamilyResult(
        spec=spec,
        study=study,
        best_pipeline=best_pipeline,
        test_roc_auc=test_roc_auc,
        preds=preds,
    )


def log_family_to_mlflow(
    result: FamilyResult,
    x_test,
    y_test,
    n_trials: int,
    cv: int,
    register_as: str | None = None,
) -> None:
    """Logger une famille de modeles dans un run MLflow imbrique.

    Parameters
    ----------
    result : FamilyResult
        Resultat a tracer (etude Optuna, pipeline, metriques).
    x_test : pandas.DataFrame
        Jeu de test, utilise pour inferer la signature et un exemple d'entree.
    y_test : array-like
        Cibles du jeu de test, utilisees pour la matrice de confusion et le
        rapport de classification.
    n_trials : int
        Nombre d'essais Optuna (loggue comme parametre).
    cv : int
        Nombre de plis de validation croisee (loggue comme parametre).
    register_as : str, optional
        Si fourni, enregistre le modele dans le Model Registry sous ce nom.
    """
    with mlflow.start_run(run_name=result.spec.name, nested=True):
        mlflow.set_tag("model_family", result.spec.name)
        mlflow.set_tag("sampler", "TPE")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv", cv)

        # TODO (S6-6) : pour chaque trial de result.study.trials, ouvrir un run
        #               MLflow imbrique (nested=True) et logger trial.params
        #               ainsi que la metrique "cv_roc_auc" = trial.value
        for trial in result.study.trials:
            ...

        mlflow.log_params(result.study.best_params)
        mlflow.log_metric("cv_roc_auc", result.study.best_value)
        mlflow.log_metric("test_roc_auc", result.test_roc_auc)

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.spec.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(str, classification_report(y_test, result.preds))
        mlflow.log_text(report_text, "classification_report.txt")

        log_shap_summary(result.best_pipeline, x_test, result.spec.name)

        signature = infer_signature(x_test, result.best_pipeline.predict(x_test))
        _model_info = mlflow.sklearn.log_model(
            result.best_pipeline,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )

        # TODO (S6-7 bonus) : renommez _model_info en model_info ; si register_as
        #               est defini et que model_info.registered_model_version est
        #               defini, appelez describe_registered_version pour
        #               documenter cette version dans le Model Registry
        #               (description + tags)


def describe_registered_version(
    name: str,
    version: int,
    result: FamilyResult,
    n_trials: int,
    cv: int,
) -> None:
    """Documenter une version enregistree dans le Model Registry.

    Ajoute une description (algorithme, hyperparametres, metriques) et des
    tags (famille de modele, methode de recherche, scores) sur la version du
    modele afin de pouvoir comparer les versions sans rouvrir le run MLflow.

    Parameters
    ----------
    name : str
        Nom du modele enregistre dans le registry.
    version : int
        Version enregistree a documenter.
    result : FamilyResult
        Resultat d'optimisation associe a cette version.
    n_trials : int
        Nombre d'essais Optuna par famille.
    cv : int
        Nombre de plis de validation croisee.
    """
    # TODO (S6-7 bonus) : creer un mlflow.MlflowClient() puis
    #   - client.update_model_version(name, version, description=...) avec un
    #     resume (famille, sampler, nombre d'essais, meilleurs hyperparametres,
    #     scores)
    #   - client.set_model_version_tag(name, version, key, value) pour des tags
    #     tels que model_family, search_method, n_trials, cv, cv_roc_auc,
    #     test_roc_auc
    raise NotImplementedError


def optimize(n_trials: int = 30, cv: int = 5, use_mlflow: bool = True) -> list[FamilyResult]:
    """Optimiser RF / XGBoost / LightGBM avec Optuna et sauvegarder le meilleur.

    Le meilleur modele (selon le ROC AUC de test) est persiste dans
    ``models/model.joblib``. Lorsque ``use_mlflow`` est actif, chaque famille
    est suivie dans un run MLflow imbrique sous un run parent
    ``optuna-compare``, et la meilleure est enregistree dans le Model Registry
    sous ``MODEL_NAME``.

    Parameters
    ----------
    n_trials : int, optional
        Nombre d'essais Optuna par famille de modeles, par defaut 30.
    cv : int, optional
        Nombre de plis de validation croisee, par defaut 5.
    use_mlflow : bool, optional
        Active le suivi MLflow, par defaut True.

    Returns
    -------
    list of FamilyResult
        Resultats tries du meilleur au moins bon (ROC AUC de test decroissant).
    """
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    if use_mlflow:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        logger.info(
            "Suivi MLflow : %s (experience: %s)", MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT
        )

    results = [
        optimize_family(spec, x_train, y_train, x_test, y_test, n_trials=n_trials, cv=cv)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.test_roc_auc, reverse=True)

    best = results[0]
    logger.info("Meilleure famille : %s (test_roc_auc=%.3f)", best.spec.name, best.test_roc_auc)

    if use_mlflow:
        with mlflow.start_run(run_name="optuna-compare"):
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv", cv)
            mlflow.set_tag("best_model", best.spec.name)
            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_family_to_mlflow(
                    result, x_test, y_test, n_trials, cv, register_as=register_as
                )
        logger.info("Meilleur modele enregistre dans le registry sous '%s'", MODEL_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_pipeline, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", MODEL_DIR / "model.joblib")

    return results


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials", type=int, default=30, help="Nombre d'essais Optuna par famille de modeles"
    )
    parser.add_argument("--cv", type=int, default=5, help="Nombre de plis de validation croisee")
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Desactive le suivi MLflow (utile sans serveur de tracking)",
    )
    args = parser.parse_args()
    optimize(n_trials=args.n_trials, cv=args.cv, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
