"""Optimisation d'hyperparametres avec Optuna."""

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
import optuna
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from optuna.samplers import TPESampler
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from mlproject.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    MODEL_NAME,
    RANDOM_STATE,
)
from mlproject.data import load_data, split
from mlproject.evaluation import log_shap_summary
from mlproject.features import build_preprocessor
from mlproject.tracking import log_dataset, setup_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelSpec:
    name: str
    suggest_params: Callable
    build_estimator: Callable[[dict], ClassifierMixin]


def build_model_specs() -> list[ModelSpec]:
    def suggest_rf(trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=50),
            "max_depth": trial.suggest_categorical("max_depth", [None, 10, 20]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 3),
        }

    def build_rf(params: dict) -> ClassifierMixin:
        return cast(
            ClassifierMixin,
            RandomForestClassifier(
                **params,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=1,
            ),
        )

    def suggest_xgb(trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
        }

    def build_xgb(params: dict) -> ClassifierMixin:
        return cast(
            ClassifierMixin,
            XGBClassifier(
                **params,
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=1,
            ),
        )

    def suggest_lgbm(trial: optuna.Trial) -> dict:
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
        }

    def build_lgbm(params: dict) -> ClassifierMixin:
        return cast(
            ClassifierMixin,
            LGBMClassifier(
                **params,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                verbose=-1,
                n_jobs=1,
            ),
        )

    return [
        ModelSpec("random_forest", suggest_rf, build_rf),
        ModelSpec("xgboost", suggest_xgb, build_xgb),
        ModelSpec("lightgbm", suggest_lgbm, build_lgbm),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", estimator),
        ]
    )


def objective(trial: optuna.Trial, spec: ModelSpec, x_train, y_train, cv: int) -> float:
    params = spec.suggest_params(trial)
    estimator = spec.build_estimator(params)
    pipeline = build_pipeline(estimator)

    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
    )

    return float(np.mean(scores))


def run_study(spec: ModelSpec, x_train, y_train, n_trials: int, cv: int) -> optuna.Study:
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=RANDOM_STATE),
        study_name=f"{spec.name}_optimization",
    )

    study.optimize(
        lambda trial: objective(trial, spec, x_train, y_train, cv),
        n_trials=n_trials,
    )

    return study


@dataclass
class FamilyResult:
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
    logger.info("Optimisation de %s avec Optuna (n_trials=%d, cv=%d)", spec.name, n_trials, cv)

    study = run_study(spec, x_train, y_train, n_trials=n_trials, cv=cv)

    best_pipeline = build_pipeline(spec.build_estimator(study.best_params))
    best_pipeline.fit(x_train, y_train)

    proba = best_pipeline.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    test_roc_auc = float(roc_auc_score(y_test, proba))

    logger.info(
        "%s : cv_roc_auc=%.3f test_roc_auc=%.3f params=%s",
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
    with mlflow.start_run(run_name=result.spec.name, nested=True):
        mlflow.set_tag("model_family", result.spec.name)
        mlflow.set_tag("search_method", "Optuna")
        mlflow.set_tag("sampler", "TPE")
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("cv", cv)

        for trial in result.study.trials:
            with mlflow.start_run(run_name=f"{result.spec.name}-trial-{trial.number}", nested=True):
                mlflow.log_params(trial.params)
                if trial.value is not None:
                    mlflow.log_metric("cv_roc_auc", float(trial.value))
                mlflow.set_tag("model_family", result.spec.name)
                mlflow.set_tag("trial_number", str(trial.number))

        mlflow.log_params(result.study.best_params)
        mlflow.log_metric("cv_roc_auc", float(result.study.best_value))
        mlflow.log_metric("test_roc_auc", result.test_roc_auc)

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.spec.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        report_text = cast(str, classification_report(y_test, result.preds))
        mlflow.log_dict(report_dict, "classification_report.json")
        mlflow.log_text(report_text, "classification_report.txt")

        log_shap_summary(result.best_pipeline, x_test, result.spec.name)

        signature = infer_signature(x_test, result.best_pipeline.predict(x_test))

        model_info = mlflow.sklearn.log_model(
            sk_model=result.best_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=x_test.iloc[:5],
            registered_model_name=register_as,
        )

        version = getattr(model_info, "registered_model_version", None)

        if register_as and version:
            describe_registered_version(
                name=register_as,
                version=int(version),
                result=result,
                n_trials=n_trials,
                cv=cv,
            )


def describe_registered_version(
    name: str,
    version: int,
    result: FamilyResult,
    n_trials: int,
    cv: int,
) -> None:
    client = mlflow.tracking.MlflowClient()

    description = (
        f"Famille : {result.spec.name} | "
        f"Methode : Optuna TPE | "
        f"n_trials={n_trials} | "
        f"cv={cv} | "
        f"Best params={result.study.best_params} | "
        f"cv_roc_auc={result.study.best_value:.3f} | "
        f"test_roc_auc={result.test_roc_auc:.3f}"
    )

    client.update_model_version(
        name=name,
        version=str(version),
        description=description,
    )

    tags = {
        "model_family": result.spec.name,
        "search_method": "Optuna",
        "sampler": "TPE",
        "n_trials": str(n_trials),
        "cv": str(cv),
        "cv_roc_auc": f"{result.study.best_value:.3f}",
        "test_roc_auc": f"{result.test_roc_auc:.3f}",
    }

    for key, value in tags.items():
        client.set_model_version_tag(name, str(version), key, value)


def optimize(n_trials: int = 10, cv: int = 2, use_mlflow: bool = True) -> list[FamilyResult]:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    if use_mlflow:
        setup_experiment()
        logger.info("Suivi MLflow : %s", MLFLOW_TRACKING_URI)
        logger.info("Experience MLflow : %s", MLFLOW_EXPERIMENT)

    results = [
        optimize_family(spec, x_train, y_train, x_test, y_test, n_trials=n_trials, cv=cv)
        for spec in build_model_specs()
    ]

    results.sort(key=lambda r: r.test_roc_auc, reverse=True)
    best = results[0]

    print("\n=== Resultats Optuna ===")
    for result in results:
        print(
            f"{result.spec.name:15s} | "
            f"cv_roc_auc={result.study.best_value:.3f} | "
            f"test_roc_auc={result.test_roc_auc:.3f}"
        )

    logger.info("Meilleure famille : %s (test_roc_auc=%.3f)", best.spec.name, best.test_roc_auc)

    if use_mlflow:
        with mlflow.start_run(run_name="optuna-compare"):
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("cv", cv)
            mlflow.set_tag("best_model", best.spec.name)

            log_dataset(
                df=df,
                context="training",
                name="telco_churn_dataset",
            )

            mlflow.log_metric("best_cv_roc_auc", float(best.study.best_value))
            mlflow.log_metric("best_test_roc_auc", best.test_roc_auc)

            for result in results:
                mlflow.log_metric(f"{result.spec.name}_cv_roc_auc", float(result.study.best_value))
                mlflow.log_metric(f"{result.spec.name}_test_roc_auc", result.test_roc_auc)

            for result in results:
                register_as = MODEL_NAME if result is best else None
                log_family_to_mlflow(
                    result=result,
                    x_test=x_test,
                    y_test=y_test,
                    n_trials=n_trials,
                    cv=cv,
                    register_as=register_as,
                )

        logger.info("Meilleur modele enregistre dans le registry sous '%s'", MODEL_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_pipeline, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", MODEL_DIR / "model.joblib")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Nombre d'essais Optuna par famille de modeles",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=2,
        help="Nombre de plis de validation croisee",
    )
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Desactive le suivi MLflow",
    )
    args = parser.parse_args()

    optimize(
        n_trials=args.n_trials,
        cv=args.cv,
        use_mlflow=args.use_mlflow,
    )


if __name__ == "__main__":
    main()