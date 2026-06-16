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
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
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

warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ModelSpec:
    name: str
    estimator: ClassifierMixin
    param_grid: dict


@dataclass
class FitResult:
    name: str
    best_estimator: Pipeline
    best_params: dict
    cv_score: float
    f1: float
    roc_auc: float
    preds: np.ndarray

def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="random_forest",
            estimator=RandomForestClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=1,
            ),
            param_grid={
                "clf__n_estimators": [50],
                "clf__max_depth": [10],
                "clf__min_samples_leaf": [2],
            },
        ),
        ModelSpec(
            name="xgboost",
            estimator=XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                n_jobs=1,
            ),
            param_grid={
                "clf__n_estimators": [50],
                "clf__max_depth": [3],
                "clf__learning_rate": [0.1],
            },
        ),
        ModelSpec(
            name="lightgbm",
            estimator=LGBMClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
                verbose=-1,
                n_jobs=1,
            ),
            param_grid={
                "clf__n_estimators": [50],
                "clf__num_leaves": [31],
                "clf__learning_rate": [0.1],
            },
        ),
    ]


def build_pipeline(estimator: ClassifierMixin) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", estimator),
        ]
    )


def optimize_model(
    spec: ModelSpec,
    x_train,
    y_train,
    x_test,
    y_test,
    cv: int = 5,
    scoring: str = "roc_auc",
) -> FitResult:
    logger.info("Optimisation de %s avec GridSearchCV (cv=%d, scoring=%s)", spec.name, cv, scoring)

    search = GridSearchCV(
        estimator=build_pipeline(spec.estimator),
        param_grid=spec.param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        refit=True,
    )

    search.fit(x_train, y_train)

    best = search.best_estimator_
    proba = best.predict_proba(x_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    return FitResult(
        name=spec.name,
        best_estimator=best,
        best_params=search.best_params_,
        cv_score=float(search.best_score_),
        f1=float(f1_score(y_test, preds)),
        roc_auc=float(roc_auc_score(y_test, proba)),
        preds=preds,
    )


def log_run_to_mlflow(
    result: FitResult,
    x_test,
    y_test,
    cv: int,
    scoring: str,
    register_as: str | None = None,
) -> None:
    with mlflow.start_run(run_name=result.name, nested=True):
        mlflow.set_tag("model_family", result.name)
        mlflow.log_param("cv", cv)
        mlflow.log_param("scoring", scoring)
        mlflow.log_params(result.best_params)

        mlflow.log_metrics(
            {
                f"cv_{scoring}": result.cv_score,
                "f1": result.f1,
                "roc_auc": result.roc_auc,
            }
        )

        cm = confusion_matrix(y_test, result.preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title(f"Matrice de confusion : {result.name}")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        report_dict = cast(dict, classification_report(y_test, result.preds, output_dict=True))
        report_text = cast(str, classification_report(y_test, result.preds))

        mlflow.log_dict(report_dict, "classification_report.json")
        mlflow.log_text(report_text, "classification_report.txt")

        log_shap_summary(result.best_estimator, x_test, result.name)

        signature = infer_signature(x_test, result.best_estimator.predict(x_test))

        model_info = mlflow.sklearn.log_model(
            sk_model=result.best_estimator,
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
                cv=cv,
                scoring=scoring,
            )


def describe_registered_version(
    name: str,
    version: int,
    result: FitResult,
    cv: int,
    scoring: str,
) -> None:
    client = mlflow.tracking.MlflowClient()

    description = (
        f"Modele : {result.name} | "
        f"Hyperparametres : {result.best_params} | "
        f"cv_{scoring}={result.cv_score:.3f} | "
        f"f1={result.f1:.3f} | "
        f"roc_auc={result.roc_auc:.3f}"
    )

    client.update_model_version(
        name=name,
        version=str(version),
        description=description,
    )

    tags = {
        "model_family": result.name,
        "search_method": "GridSearchCV",
        "cv": str(cv),
        "scoring": scoring,
        "f1": f"{result.f1:.3f}",
        "roc_auc": f"{result.roc_auc:.3f}",
    }

    for key, value in tags.items():
        client.set_model_version_tag(name, str(version), key, value)


def train_all(
    cv: int = 5,
    scoring: str = "roc_auc",
    use_mlflow: bool = True,
) -> list[FitResult]:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    if use_mlflow:
        setup_experiment()
        logger.info("Suivi MLflow : %s", MLFLOW_TRACKING_URI)
        logger.info("Experience MLflow : %s", MLFLOW_EXPERIMENT)

    results = [
        optimize_model(
            spec=spec,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            cv=cv,
            scoring=scoring,
        )
        for spec in build_model_specs()
    ]

    results.sort(key=lambda r: r.roc_auc, reverse=True)

    best = results[0]

    print("\n=== Resultats des modeles ===")
    for result in results:
        print(
            f"{result.name:15s} | "
            f"cv_score={result.cv_score:.3f} | "
            f"f1={result.f1:.3f} | "
            f"roc_auc={result.roc_auc:.3f}"
        )

    logger.info("Meilleur modele : %s (roc_auc=%.3f)", best.name, best.roc_auc)

    if use_mlflow:
        with mlflow.start_run(run_name="compare-models"):
            mlflow.log_param("cv", cv)
            mlflow.log_param("scoring", scoring)
            mlflow.set_tag("best_model", best.name)

            log_dataset(
                df=df,
                context="training",
                name="telco_churn_dataset",
            )

            mlflow.log_metric("best_f1", best.f1)
            mlflow.log_metric("best_roc_auc", best.roc_auc)
            mlflow.log_metric(f"best_cv_{scoring}", best.cv_score)

            for result in results:
                mlflow.log_metric(f"{result.name}_f1", result.f1)
                mlflow.log_metric(f"{result.name}_roc_auc", result.roc_auc)
                mlflow.log_metric(f"{result.name}_cv_{scoring}", result.cv_score)

            for result in results:
                register_as = MODEL_NAME if result.name == best.name else None
                log_run_to_mlflow(
                    result=result,
                    x_test=x_test,
                    y_test=y_test,
                    cv=cv,
                    scoring=scoring,
                    register_as=register_as,
                )

        logger.info("Meilleur modele enregistre dans le registry sous '%s'", MODEL_NAME)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best.best_estimator, MODEL_DIR / "model.joblib")
    logger.info("Modele sauvegarde dans %s", MODEL_DIR / "model.joblib")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--scoring", type=str, default="roc_auc")
    parser.add_argument("--no-mlflow", dest="use_mlflow", action="store_false")
    args = parser.parse_args()

    train_all(
        cv=args.cv,
        scoring=args.scoring,
        use_mlflow=args.use_mlflow,
    )


if __name__ == "__main__":
    main()