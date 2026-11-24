"""Chargement du modèle « champion » : depuis le bundle local ou depuis le Model Registry MLflow.

MODEL_SOURCE=local   (défaut) : lit MODEL_PATH (models/model.joblib)
MODEL_SOURCE=mlflow           : lit models:/<nom>@champion depuis MLFLOW_TRACKING_URI
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np
import pandas as pd

from churn.config import load_config, resolve

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    pipeline: Any
    threshold: float
    name: str
    version: str
    source: str
    trained_at: str | None = None
    test_metrics: dict[str, float] = field(default_factory=dict)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]


def _load_local(path: str) -> LoadedModel:
    # Le fichier est produit par notre propre pipeline, le charger avec joblib est sûr.
    bundle = joblib.load(path)  # noqa: S301
    return LoadedModel(
        pipeline=bundle["pipeline"],
        threshold=float(bundle["threshold"]),
        name=bundle["model_name"],
        version=str(bundle["model_version"]),
        source=f"local:{path}",
        trained_at=bundle.get("trained_at"),
        test_metrics={k: float(v) for k, v in bundle.get("test_metrics", {}).items()},
    )


def _load_mlflow(cfg: dict[str, Any]) -> LoadedModel:
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    name = cfg["mlflow"]["registered_model_name"]
    client = MlflowClient()
    mv = client.get_model_version_by_alias(name, "champion")
    # Les types de confiance déclarés à l'enregistrement sont relus depuis le modèle
    pipeline = mlflow.sklearn.load_model(f"models:/{name}@champion")
    run = client.get_run(mv.run_id)
    test_metrics = {k.removeprefix("test_"): v for k, v in run.data.metrics.items()
                    if k.startswith("test_")}
    return LoadedModel(
        pipeline=pipeline,
        threshold=float(mv.tags.get("decision_threshold", 0.5)),
        name=run.data.params.get("best_model", name),
        version=str(mv.version),
        source=f"mlflow:models:/{name}@champion",
        test_metrics=test_metrics,
    )


class ModelStore:
    """Conteneur thread-safe : permet de recharger le modèle à chaud sans redémarrer l'API."""

    def __init__(self) -> None:
        self._model: LoadedModel | None = None
        self._lock = threading.Lock()

    @property
    def model(self) -> LoadedModel | None:
        return self._model

    def load(self) -> LoadedModel:
        cfg = load_config()
        source = os.getenv("MODEL_SOURCE", "local")
        if source == "mlflow":
            model = _load_mlflow(cfg)
        else:
            path = os.getenv("MODEL_PATH", str(resolve(cfg["artifacts"]["model_dir"]) / "model.joblib"))
            model = _load_local(path)
        with self._lock:
            self._model = model
        logger.info("Modèle chargé : %s v%s (%s)", model.name, model.version, model.source)
        return model
