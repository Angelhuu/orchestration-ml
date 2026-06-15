"""Entrainement et optimisation de plusieurs modeles de classification (AutoML + SHAP).

Seance 7 - TP AutoML & SHAP
    Ce module compare trois familles de modeles (Random Forest, XGBoost,
    LightGBM), chacune optimisee par recherche d'hyperparametres en grille
    (GridSearchCV), et persiste la meilleure dans `models/model.joblib`.
    Completez les TODO (S7-n).

Chaque modele est suivi dans MLflow (un run par modele, imbrique sous un run
parent ``compare-models``) et le meilleur est enregistre dans le Model
Registry, avec une description et des tags (TODO S7-5, bonus) et un summary
plot SHAP loggue comme artefact (`mlproject.evaluation.log_shap_summary`, deja
fourni).

Lancement :
    python -m mlproject.train_models
    python -m mlproject.train_models --cv 3 --scoring roc_auc
    python -m mlproject.train_models --no-mlflow   # desactive le suivi MLflow
"""
from __future__ import annotations

import argparse
import logging
import warnings
from dataclasses import dataclass
from typing import cast

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
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

# TODO (S7-1) : importer RandomForestClassifier (sklearn.ensemble),
#               XGBClassifier (xgboost), LGBMClassifier (lightgbm),
#               GridSearchCV (sklearn.model_selection) et RANDOM_STATE
#               (mlproject.config)

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

# Le ColumnTransformer renvoie un tableau numpy sans noms de colonnes lors du
# scoring interne de la validation croisee : on neutralise l'avertissement
# correspondant, sans incidence sur les predictions.
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


@dataclass
class ModelSpec:
    """Specification d'un modele a optimiser.

    Parameters
    ----------
    name : str
        Identifiant lisible du modele.
    estimator : ClassifierMixin
        Instance du classifieur scikit-learn compatible.
    param_grid : dict
        Grille d'hyperparametres pour ``GridSearchCV``. Les cles sont
        prefixees par ``clf__`` car le classifieur est la derniere etape
        du pipeline.
    """

    name: str
    estimator: ClassifierMixin
    param_grid: dict


def build_model_specs() -> list[ModelSpec]:
    """Construire la liste des modeles a optimiser.

    Returns
    -------
    list of ModelSpec
        Random Forest, XGBoost et LightGBM avec leurs grilles respectives.
    """
    # TODO (S7-2) : retourner une liste de trois ModelSpec :
    #   - random_forest : RandomForestClassifier(random_state=RANDOM_STATE),
    #     grille clf__n_estimators=[100, 200], clf__max_depth=[None, 10, 20],
    #     clf__min_samples_leaf=[1, 2]
    #   - xgboost : XGBClassifier(random_state=RANDOM_STATE,
    #     eval_metric="logloss", n_jobs=-1), grille clf__n_estimators=[100, 200],
    #     clf__max_depth=[3, 5], clf__learning_rate=[0.1, 0.01]
    #   - lightgbm : LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
    #     grille clf__n_estimators=[100, 200], clf__num_leaves=[31, 63],
    #     clf__learning_rate=[0.1, 0.01]
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


@dataclass
class FitResult:
    """Resultat d'optimisation d'un modele.

    Parameters
    ----------
    name : str
        Identifiant du modele.
    best_estimator : Pipeline
        Pipeline reentraine avec les meilleurs hyperparametres.
    best_params : dict
        Meilleurs hyperparametres trouves par la recherche.
    cv_score : float
        Meilleur score moyen de validation croisee.
    f1 : float
        F1-score sur le jeu de test.
    roc_auc : float
        ROC AUC sur le jeu de test.
    preds : np.ndarray
        Predictions (classes) sur le jeu de test.
    """

    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    f1: float
    roc_auc: float
    preds: np.ndarray


def optimize_model(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> FitResult:
    """Optimiser un modele par GridSearchCV et l'evaluer sur le test.

    Parameters
    ----------
    spec : ModelSpec
        Modele et grille d'hyperparametres.
    x_train, y_train : array-like
        Donnees d'entrainement.
    x_test, y_test : array-like
        Donnees de test pour l'evaluation finale.
    cv : int, optional
        Nombre de plis de validation croisee, par defaut 5.
    scoring : str, optional
        Metrique optimisee par la recherche, par defaut "roc_auc".

    Returns
    -------
    FitResult
        Meilleur estimateur et metriques associees.
    """
    logger.info("Optimisation de %s (cv=%d, scoring=%s)", spec.name, cv, scoring)

    # TODO (S7-3) :
    #   - construire un GridSearchCV(estimator=build_pipeline(spec.estimator),
    #     param_grid=spec.param_grid, cv=cv, scoring=scoring, n_jobs=-1, refit=True)
    #   - l'entrainer avec search.fit(x_train, y_train)
    #   - recuperer best = search.best_estimator_
    #   - calculer proba = best.predict_proba(x_test)[:, 1] et
    #     preds = (proba >= 0.5).astype(int)
    #   - retourner un FitResult avec best_params=search.best_params_,
    #     cv_score=float(search.best_score_), f1=f1_score(y_test, preds),
    #     roc_auc=roc_auc_score(y_test, proba)
    raise NotImplementedError


def log_run_to_mlflow(
    result: FitResult,
    x_test,
    y_test,
    cv: int,
    scoring: str,
    register_as: str | None = None,
) -> None:
    """Logger un resultat d'optimisation dans un run MLflow imbrique.

    Parameters
    ----------
    result : FitResult
        Resultat a tracer (params, metriques, estimateur).
    x_test : pandas.DataFrame
        Jeu de test, utilise pour inferer la signature et un exemple d'entree.
    y_test : array-like
        Cibles du jeu de test, utilisees pour la matrice de confusion et le
        rapport de classification.
    cv : int
        Nombre de plis de validation croisee (loggue comme parametre).
    scoring : str
        Metrique optimisee (prefixe le nom de la metrique de CV loggee).
    register_as : str, optional
        Si fourni, enregistre le modele dans le Model Registry sous ce nom.
    """
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)
        mlflow.log_param("cv", cv)
        mlflow.log_param("scoring", scoring)

        # TODO (S7-4a) : logger les hyperparametres (mlflow.log_params(result.best_params))
        #               et les metriques cv_<scoring>, f1, roc_auc
        #               (mlflow.log_metrics({...}))

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        mlflow.log_dict(report_dict, "classification_report.json")
        report_text = cast(str, classification_report(y_test, result.preds))
        mlflow.log_text(report_text, "classification_report.txt")

        # TODO (S7-4b) : appeler log_shap_summary(result.best_estimator, x_test,
        #               result.name) pour logger le summary plot SHAP
        #               (artefact "shap_summary.png")

        signature = infer_signature(x_test, result.best_estimator.predict(x_test))
        model_info = mlflow.sklearn.log_model(
            result.best_estimator,
            name="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )

        # TODO (S7-5 bonus) : si register_as est defini et que
        #               model_info.registered_model_version est defini, appeler
        #               describe_registered_version pour documenter cette
        #               version dans le Model Registry (description + tags)


def describe_registered_version(
    name: str,
    version: int,
    result: FitResult,
    cv: int,
    scoring: str,
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
    result : FitResult
        Resultat d'optimisation associe a cette version.
    cv : int
        Nombre de plis de validation croisee utilise pour l'optimisation.
    scoring : str
        Metrique optimisee par GridSearchCV.
    """
    # TODO (S7-5 bonus) : creer un mlflow.MlflowClient() puis
    #   - client.update_model_version(name, str(version), description=...) avec
    #     un resume (famille, hyperparametres, metriques de test)
    #   - client.set_model_version_tag(name, str(version), key, value) pour des
    #     tags tels que model_family, search_method, cv, scoring, f1, roc_auc
    raise NotImplementedError


def train_all(
    cv: int = 5,
    scoring: str = "roc_auc",
    use_mlflow: bool = True,
) -> list[FitResult]:
    """Entrainer et comparer les trois modeles, sauvegarder le meilleur.

    Le meilleur modele (selon le ROC AUC de test) est persiste dans
    ``models/model.joblib`` afin de rester compatible avec l'API d'inference.
    Lorsque ``use_mlflow`` est actif, chaque modele est suivi dans un run
    MLflow imbrique sous un run parent ``compare-models``, et le meilleur est
    enregistre dans le Model Registry sous ``MODEL_NAME``.

    Parameters
    ----------
    cv : int, optional
        Nombre de plis de validation croisee, par defaut 5.
    scoring : str, optional
        Metrique optimisee par GridSearchCV, par defaut "roc_auc".
    use_mlflow : bool, optional
        Active le suivi MLflow, par defaut True. Necessite un serveur de
        tracking accessible (voir MLFLOW_TRACKING_URI).

    Returns
    -------
    list of FitResult
        Resultats tries du meilleur au moins bon (ROC AUC decroissant).
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
        optimize_model(spec, x_train, y_train, x_test, y_test, cv=cv, scoring=scoring)
        for spec in build_model_specs()
    ]
    results.sort(key=lambda r: r.roc_auc, reverse=True)

    best = results[0]
    logger.info("Meilleur modele : %s (roc_auc=%.3f)", best.name, best.roc_auc)

    if use_mlflow:
        with mlflow.start_run(run_name="compare-models"):
            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)
            mlflow.set_tag("best_model", best.name)
            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_run_to_mlflow(result, x_test, y_test, cv, scoring, register_as=register_as)
        logger.info("Meilleur modele enregistre dans le registry sous '%s'", MODEL_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_estimator, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", MODEL_DIR / "model.joblib")

    return results


def main() -> None:
    """Point d'entree en ligne de commande."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5, help="Nombre de plis de validation croisee")
    parser.add_argument(
        "--scoring",
        type=str,
        default="roc_auc",
        help="Metrique optimisee par GridSearchCV (ex: roc_auc, f1, accuracy)",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Desactive le suivi MLflow (utile sans serveur de tracking)",
    )
    args = parser.parse_args()
    train_all(cv=args.cv, scoring=args.scoring, use_mlflow=args.use_mlflow)


if __name__ == "__main__":
    main()
