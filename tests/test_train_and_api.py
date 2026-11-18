"""Tests d'intégration : entraînement réel (MLflow) puis API servie avec ce modèle."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "customer.json").read_text())


def test_training_promotes_model(trained_env):
    s = trained_env["summary"]
    models = Path(trained_env["cfg"]["artifacts"]["model_dir"])
    assert s["quality_gate"]["passed"], s["quality_gate"]["failures"]
    assert s["model_version"] is not None
    assert set(s["candidates"]) == {"logistic_regression", "random_forest", "lightgbm"}
    for f in ("model.joblib", "metrics.json", "reference.parquet", "plots/roc_curve.png"):
        assert (models / f).exists(), f


def test_model_registered_with_champion_alias(trained_env):
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=trained_env["cfg"]["mlflow"]["tracking_uri"])
    mv = client.get_model_version_by_alias("churn-classifier", "champion")
    assert str(mv.version) == str(trained_env["summary"]["model_version"])
    assert float(mv.tags["decision_threshold"]) == trained_env["summary"]["threshold"]


def test_failed_quality_gate_does_not_promote(trained_env, tmp_path):
    import yaml

    from churn.config import load_config
    from churn.train import run

    cfg = dict(trained_env["cfg"])
    cfg["quality_gate"] = {"min_roc_auc": 0.999}
    cfg["artifacts"] = {"model_dir": str(tmp_path), "reference_path": str(tmp_path / "ref.parquet")}
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg))
    summary = run(str(path))
    load_config.cache_clear()
    assert not summary["quality_gate"]["passed"]
    assert not (tmp_path / "model.joblib").exists()


@pytest.fixture(scope="module")
def client(trained_env):
    from churn.api.main import app

    with TestClient(app) as c:
        yield c


def test_health_and_info(client, trained_env):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True,
                                            "version": "1.0.0"}
    info = client.get("/model/info").json()
    assert info["model_version"] == str(trained_env["summary"]["model_version"])
    assert info["threshold"] == trained_env["summary"]["threshold"]


def test_predict_single(client):
    r = client.post("/predict", json=FIXTURE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["will_churn"] == (body["churn_probability"] >= body["threshold"])
    assert body["risk_level"] in {"low", "medium", "high"}


def test_predict_rejects_invalid_payload(client):
    bad = {**FIXTURE, "Contract": "Lifetime", "tenure": -3}
    r = client.post("/predict", json=bad)
    assert r.status_code == 422
    assert {e["loc"][-1] for e in r.json()["detail"]} == {"Contract", "tenure"}


def test_batch_and_prediction_log(client, trained_env):
    r = client.post("/predict/batch", json={"customers": [FIXTURE] * 60})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 60
    log = Path(trained_env["cfg"]["monitoring"]["predictions_log"])
    lines = log.read_text().strip().splitlines()
    assert len(lines) >= 60
    assert "prediction_proba" in json.loads(lines[-1])


def test_drift_endpoint_and_metrics(client):
    r = client.get("/monitoring/drift")
    assert r.status_code == 200
    # 60 fois le même client → distribution très différente de l'entraînement
    assert r.json()["retrain_recommended"] is True
    metrics = client.get("/metrics").text
    assert "churn_predictions_total" in metrics
    assert "churn_feature_drift_share" in metrics
    assert 'churn_model_info{name=' in metrics


def test_reload_model(client):
    r = client.post("/model/reload")
    assert r.status_code == 200 and r.json()["model_name"]
