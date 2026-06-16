"""Entrainement du modele de classification baseline avec MLflow."""

from __future__ import annotations

import argparse

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from mlproject.config import MODEL_DIR
from mlproject.data import load_data, split
from mlproject.features import build_preprocessor
from mlproject.tracking import log_dataset, setup_experiment


def build_model(c: float = 1.0, max_iter: int = 1000) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("clf", LogisticRegression(C=c, max_iter=max_iter)),
        ]
    )


def train(c: float = 1.0, max_iter: int = 1000) -> dict:
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)

    setup_experiment()

    with mlflow.start_run(run_name="baseline-logistic-regression"):
        mlflow.log_params(
            {
                "model_type": "logistic_regression",
                "c": c,
                "max_iter": max_iter,
            }
        )

        log_dataset(df=df, context="training", name="telco_churn_dataset")

        model = build_model(c=c, max_iter=max_iter)
        model.fit(x_train, y_train)

        proba = model.predict_proba(x_test)[:, 1]
        preds = (proba >= 0.5).astype(int)

        metrics = {
            "f1": float(f1_score(y_test, preds)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }

        print(f"f1={metrics['f1']:.3f}  roc_auc={metrics['roc_auc']:.3f}")

        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, preds)
        fig, ax = plt.subplots(figsize=(5, 5))
        ConfusionMatrixDisplay(cm).plot(ax=ax)
        ax.set_title("Matrice de confusion : baseline")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        signature = infer_signature(x_test, model.predict(x_test))

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=x_test.iloc[:5],
        )

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_DIR / "model.joblib")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    train(c=args.c, max_iter=args.max_iter)


if __name__ == "__main__":
    main()