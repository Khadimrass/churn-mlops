"""API de serving du modèle de churn.

Lancer : uvicorn churn.api.main:app --host 0.0.0.0 --port 8000
Docs interactives : http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from churn import __version__
from churn.api.model_store import LoadedModel, ModelStore
from churn.api.observability import (
    DRIFT_SHARE,
    LATENCY,
    MODEL_INFO,
    PREDICTIONS,
    PROBABILITY,
    REQUESTS,
    PredictionLogger,
    risk_level,
)
from churn.api.schemas import (
    BatchRequest,
    BatchResponse,
    Customer,
    Health,
    ModelInfo,
    Prediction,
    PredictionResponse,
)
from churn.config import load_config, resolve
from churn.features import to_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("churn.api")

store = ModelStore()


def _set_model_gauge(model: LoadedModel) -> None:
    MODEL_INFO.clear()
    MODEL_INFO.labels(name=model.name, version=model.version).set(1)


async def _periodic_drift_check(interval: int) -> None:
    """Tâche de fond : recalcule la dérive régulièrement pour que Grafana/Prometheus la suivent."""
    from churn.monitoring.drift import run as run_drift

    while True:
        await asyncio.sleep(interval)
        try:
            report = await asyncio.to_thread(run_drift, None, 1000)
            if report is not None:
                DRIFT_SHARE.set(report.drift_share)
        except Exception:
            logger.exception("Échec du calcul de dérive périodique")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    log_path = os.getenv("PREDICTIONS_LOG", str(resolve(cfg["monitoring"]["predictions_log"])))
    app.state.pred_logger = PredictionLogger(log_path)
    try:
        _set_model_gauge(store.load())
    except Exception:  # l'API démarre quand même, /health signalera « degraded »
        logger.exception("Impossible de charger le modèle au démarrage")

    task = None
    if (interval := int(os.getenv("DRIFT_CHECK_INTERVAL_SECONDS", "0"))) > 0:
        task = asyncio.create_task(_periodic_drift_check(interval))
    yield
    if task:
        task.cancel()


app = FastAPI(
    title="Churn Prediction API",
    description="Prédit la probabilité qu'un client résilie son abonnement télécom.",
    version=__version__,
    lifespan=lifespan,
)


@app.middleware("http")
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    if request.url.path != "/metrics":
        REQUESTS.labels(endpoint=request.url.path, status=str(response.status_code)).inc()
    return response


def _require_model() -> LoadedModel:
    if store.model is None:
        raise HTTPException(status_code=503, detail="Aucun modèle chargé")
    return store.model


def _predict(customers: list[Customer], endpoint: str) -> tuple[LoadedModel, list[Prediction]]:
    model = _require_model()
    records = [c.model_dump() for c in customers]
    start = time.perf_counter()
    probas = model.predict_proba(to_frame(records))
    LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)

    preds = []
    for p in probas:
        will_churn = bool(p >= model.threshold)
        PROBABILITY.observe(float(p))
        PREDICTIONS.labels(outcome="churn" if will_churn else "stay").inc()
        preds.append(Prediction(churn_probability=round(float(p), 4), will_churn=will_churn,
                                risk_level=risk_level(float(p))))
    app.state.pred_logger.log(records, probas.tolist(), model.version)
    return model, preds


@app.get("/health", response_model=Health, tags=["ops"])
def health() -> Health:
    loaded = store.model is not None
    return Health(status="ok" if loaded else "degraded", model_loaded=loaded, version=__version__)


@app.get("/model/info", response_model=ModelInfo, tags=["ops"])
def model_info() -> ModelInfo:
    m = _require_model()
    return ModelInfo(model_name=m.name, model_version=m.version, source=m.source,
                     threshold=m.threshold, trained_at=m.trained_at, test_metrics=m.test_metrics)


@app.post("/model/reload", response_model=ModelInfo, tags=["ops"])
def reload_model() -> ModelInfo:
    """Recharge le modèle champion (après un nouvel entraînement) sans redémarrer le service."""
    try:
        _set_model_gauge(store.load())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rechargement échoué : {exc}") from exc
    return model_info()


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(customer: Customer) -> PredictionResponse:
    model, [pred] = _predict([customer], "/predict")
    return PredictionResponse(**pred.model_dump(), model_version=model.version,
                              threshold=model.threshold)


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(batch: BatchRequest) -> BatchResponse:
    model, preds = _predict(batch.customers, "/predict/batch")
    return BatchResponse(predictions=preds, model_version=model.version,
                         threshold=model.threshold,
                         n_high_risk=sum(p.risk_level == "high" for p in preds))


@app.get("/monitoring/drift", tags=["ops"])
def drift(last_n: int = 1000) -> dict:
    """Calcule la dérive sur les N dernières prédictions et met à jour la jauge Prometheus."""
    from churn.monitoring.drift import run as run_drift

    report = run_drift(last_n=last_n)
    if report is None:
        raise HTTPException(status_code=409, detail="Pas encore assez de prédictions journalisées")
    DRIFT_SHARE.set(report.drift_share)
    return {
        "drift_share": report.drift_share,
        "dataset_drift": report.dataset_drift,
        "prediction_drift": report.prediction_drift,
        "retrain_recommended": report.retrain_recommended,
        "n_current": report.n_current,
        "drifted_features": [f.feature for f in report.features if f.drifted],
    }


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
