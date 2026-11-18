"""Métriques Prometheus + journal des prédictions (source du monitoring de dérive)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from prometheus_client import Counter, Gauge, Histogram

REQUESTS = Counter(
    "churn_api_requests_total", "Requêtes HTTP reçues", ["endpoint", "status"]
)
PREDICTIONS = Counter(
    "churn_predictions_total", "Prédictions émises", ["outcome"]
)
LATENCY = Histogram(
    "churn_inference_latency_seconds", "Temps d'inférence du modèle", ["endpoint"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)
PROBABILITY = Histogram(
    "churn_probability", "Distribution des probabilités de churn prédites",
    buckets=[i / 10 for i in range(1, 11)],
)
MODEL_INFO = Gauge(
    "churn_model_info", "Modèle actuellement servi (valeur toujours 1)", ["name", "version"]
)
DRIFT_SHARE = Gauge(
    "churn_feature_drift_share", "Part des features en dérive (dernier rapport)"
)


def risk_level(p: float) -> str:
    if p >= 0.70:
        return "high"
    if p >= 0.40:
        return "medium"
    return "low"


class PredictionLogger:
    """Ajoute chaque prédiction dans un fichier JSONL (une ligne = un client)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, records: list[dict[str, Any]], probas: list[float], model_version: str) -> None:
        ts = datetime.now(timezone.utc).isoformat()
        lines = [
            json.dumps({**rec, "prediction_proba": round(float(p), 6),
                        "model_version": model_version, "timestamp": ts})
            for rec, p in zip(records, probas, strict=True)
        ]
        with self._lock, open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
