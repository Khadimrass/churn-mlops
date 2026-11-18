"""Pipeline d'entraînement complet, tracé dans MLflow.

Étapes :
  1. chargement + validation des données
  2. entraînement de plusieurs modèles candidats (un run MLflow imbriqué chacun)
  3. sélection du meilleur sur la validation + choix du seuil de décision
  4. évaluation finale sur le test + quality gate
  5. si le gate passe : enregistrement dans le Model Registry (alias « champion »)
     et export du bundle local utilisé par l'API

Usage : python -m churn.train [--config chemin.yaml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay  # noqa: E402

from churn import __version__  # noqa: E402
from churn.config import load_config, resolve  # noqa: E402
from churn.data import load_dataset, split_dataset  # noqa: E402
from churn.evaluate import best_threshold, compute_metrics, quality_gate  # noqa: E402
from churn.features import build_pipeline  # noqa: E402
from churn.validation import validate  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("churn.train")


def _plot_evaluation(y_true, proba, threshold: float, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, (proba >= threshold).astype(int), display_labels=["Reste", "Part"],
        cmap="Blues", ax=ax,
    )
    ax.set_title(f"Matrice de confusion (seuil = {threshold})")
    paths.append(out_dir / "confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_title("Courbe ROC — jeu de test")
    paths.append(out_dir / "roc_curve.png")
    fig.tight_layout()
    fig.savefig(paths[-1], dpi=120)
    plt.close(fig)
    return paths


def _feature_importance(pipeline) -> pd.DataFrame | None:
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return None
    return (
        pd.DataFrame({"feature": names, "importance": values})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def train_candidates(cfg, X_train, y_train, X_val, y_val) -> list[dict[str, Any]]:
    results = []
    for name, params in cfg["models"].items():
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.set_tag("stage", "candidate")
            mlflow.log_params({f"{name}.{k}": v for k, v in params.items()})
            start = time.perf_counter()
            pipe = build_pipeline(name, params, cfg).fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start

            proba = pipe.predict_proba(X_val)[:, 1]
            metrics = compute_metrics(y_val, proba)
            mlflow.log_metrics({f"val_{k}": v for k, v in metrics.items()})
            mlflow.log_metric("fit_seconds", fit_seconds)
            logger.info("%-20s val ROC-AUC = %.4f | PR-AUC = %.4f | %.1fs",
                        name, metrics["roc_auc"], metrics["pr_auc"], fit_seconds)
            results.append({"name": name, "pipeline": pipe, "val_proba": proba,
                            "val_metrics": metrics})
    return results


def run(config_path: str | None = None) -> dict[str, Any]:
    cfg = load_config(config_path)
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    # 1. Données
    df = load_dataset(cfg)
    report = validate(df, cfg)
    for w in report.warnings:
        logger.warning(w)
    report.raise_if_failed()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(df, cfg)

    with mlflow.start_run(run_name=f"training-{datetime.now():%Y%m%d-%H%M%S}") as parent:
        mlflow.set_tags({"stage": "pipeline", "code_version": __version__})
        mlflow.log_params({
            "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
            "churn_rate_train": round(float(y_train.mean()), 4),
        })

        # 2-3. Candidats + sélection
        results = train_candidates(cfg, X_train, y_train, X_val, y_val)
        best = max(results, key=lambda r: r["val_metrics"]["roc_auc"])
        threshold = best_threshold(y_val, best["val_proba"])
        logger.info("Meilleur modèle : %s (seuil optimal F1 = %.2f)", best["name"], threshold)

        # 4. Évaluation finale
        pipe = best["pipeline"]
        test_proba = pipe.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, test_proba, threshold)
        passed, failures = quality_gate(test_metrics, cfg["quality_gate"])

        mlflow.log_params({"best_model": best["name"], "decision_threshold": threshold})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.set_tag("quality_gate", "passed" if passed else "failed")

        model_dir = resolve(cfg["artifacts"]["model_dir"])
        for p in _plot_evaluation(y_test, test_proba, threshold, model_dir / "plots"):
            mlflow.log_artifact(str(p), artifact_path="plots")
        importance = _feature_importance(pipe)
        if importance is not None:
            imp_path = model_dir / "feature_importance.csv"
            importance.to_csv(imp_path, index=False)
            mlflow.log_artifact(str(imp_path))

        summary = {
            "run_id": parent.info.run_id,
            "best_model": best["name"],
            "threshold": threshold,
            "candidates": {r["name"]: r["val_metrics"]["roc_auc"] for r in results},
            "test_metrics": test_metrics,
            "quality_gate": {"passed": passed, "failures": failures},
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_version": None,
        }

        if not passed:
            logger.error("Quality gate ÉCHOUÉ : %s", "; ".join(failures))
            _write_json(model_dir / "metrics.json", summary)
            return summary

        # 5. Promotion : registry MLflow + bundle local pour l'API
        signature = infer_signature(X_test.head(50), pipe.predict_proba(X_test.head(50))[:, 1])
        info = mlflow.sklearn.log_model(
            pipe,
            name="model",
            signature=signature,
            input_example=X_test.head(3),
            registered_model_name=cfg["mlflow"]["registered_model_name"],
            # MLflow sérialise avec skops (format sûr) : on déclare explicitement nos types
            skops_trusted_types=["churn.features.add_engineered_features", "numpy.dtype"],
        )
        version = info.registered_model_version
        client = MlflowClient()
        client.set_registered_model_alias(cfg["mlflow"]["registered_model_name"], "champion", version)
        client.set_model_version_tag(cfg["mlflow"]["registered_model_name"], version,
                                     "decision_threshold", str(threshold))
        summary["model_version"] = version

        bundle = {
            "pipeline": pipe,
            "threshold": threshold,
            "model_name": best["name"],
            "model_version": str(version),
            "run_id": parent.info.run_id,
            "trained_at": summary["trained_at"],
            "test_metrics": test_metrics,
        }
        joblib.dump(bundle, model_dir / "model.joblib")

        # Données de référence pour le monitoring de dérive
        reference = X_train.copy()
        reference["prediction_proba"] = pipe.predict_proba(X_train)[:, 1]
        reference.to_parquet(resolve(cfg["artifacts"]["reference_path"]), index=False)

        _write_json(model_dir / "metrics.json", summary)
        mlflow.log_artifact(str(model_dir / "metrics.json"))
        logger.info("✅ Quality gate OK — modèle v%s promu « champion » | test ROC-AUC = %.4f "
                    "| F1 = %.4f | Recall = %.4f", version, test_metrics["roc_auc"],
                    test_metrics["f1"], test_metrics["recall"])
    return summary


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne et promeut le modèle de churn.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    summary = run(args.config)
    sys.exit(0 if summary["quality_gate"]["passed"] else 1)


if __name__ == "__main__":
    main()
