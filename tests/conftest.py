"""Fixtures partagées : données synthétiques réalistes + environnement entraîné isolé."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
import yaml

from churn.config import DEFAULT_CONFIG, load_config
from churn.validation import ALLOWED_VALUES

os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"


def make_customers(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """Génère des clients synthétiques respectant le contrat de données,
    avec un churn corrélé aux vrais facteurs (contrat mensuel, faible ancienneté, fibre)."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({col: rng.choice(sorted(vals), n) for col, vals in ALLOWED_VALUES.items()})
    df["customerID"] = [f"C{i:05d}" for i in range(n)]
    df["SeniorCitizen"] = rng.integers(0, 2, n)
    df["tenure"] = rng.integers(0, 73, n)
    df["MonthlyCharges"] = rng.uniform(18, 120, n).round(2)
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2)
    logit = (
        -1.2
        + 1.6 * (df["Contract"] == "Month-to-month")
        + 0.8 * (df["InternetService"] == "Fiber optic")
        - 0.04 * df["tenure"]
        + 0.5 * (df["PaymentMethod"] == "Electronic check")
    )
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df


@pytest.fixture(scope="session")
def base_cfg() -> dict:
    return load_config(DEFAULT_CONFIG)


@pytest.fixture
def customers() -> pd.DataFrame:
    from churn.data import clean

    return clean(make_customers(), load_config(DEFAULT_CONFIG))


@pytest.fixture(scope="session")
def trained_env(tmp_path_factory):
    """Entraîne un vrai modèle dans un dossier temporaire et pointe la config dessus."""
    root = tmp_path_factory.mktemp("churn")
    raw = root / "raw.csv"
    make_customers(2000, seed=1).to_csv(raw, index=False)

    cfg = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cfg["data"]["raw_path"] = str(raw)
    cfg["mlflow"]["tracking_uri"] = f"sqlite:///{root / 'mlflow.db'}"
    cfg["artifacts"] = {"model_dir": str(root / "models"),
                        "reference_path": str(root / "models" / "reference.parquet")}
    cfg["monitoring"]["predictions_log"] = str(root / "prod" / "predictions.jsonl")
    cfg["monitoring"]["reports_dir"] = str(root / "reports")
    cfg["monitoring"]["min_samples"] = 50
    cfg["quality_gate"] = {"min_roc_auc": 0.70, "min_f1": 0.40, "min_recall": 0.40}
    # Modèles plus légers pour des tests rapides
    cfg["models"]["random_forest"]["n_estimators"] = 50
    cfg["models"]["lightgbm"]["n_estimators"] = 50
    (root / "models").mkdir()
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    old = {k: os.environ.get(k) for k in ("CHURN_CONFIG", "MLFLOW_TRACKING_URI", "MODEL_SOURCE")}
    os.environ["CHURN_CONFIG"] = str(config_path)
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ["MODEL_SOURCE"] = "local"
    load_config.cache_clear()

    from churn.train import run

    summary = run(str(config_path))
    yield {"root": root, "config_path": config_path, "cfg": cfg, "summary": summary}

    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    load_config.cache_clear()
