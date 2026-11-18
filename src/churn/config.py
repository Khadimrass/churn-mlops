"""Chargement de la configuration YAML (un seul point d'entrée pour tous les modules)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Racine du projet : surchargeable (ex. /app dans Docker) via CHURN_HOME
PROJECT_ROOT = Path(os.getenv("CHURN_HOME", Path(__file__).resolve().parents[2]))
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Lit le fichier de config. Le chemin peut venir de la variable CHURN_CONFIG."""
    config_path = Path(path or os.getenv("CHURN_CONFIG", DEFAULT_CONFIG))
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Les variables d'environnement l'emportent sur le fichier (utile en Docker/CI)
    if uri := os.getenv("MLFLOW_TRACKING_URI"):
        cfg["mlflow"]["tracking_uri"] = uri
    return cfg


def resolve(path: str | os.PathLike) -> Path:
    """Transforme un chemin relatif de la config en chemin absolu depuis la racine du projet."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def feature_columns(cfg: dict[str, Any]) -> list[str]:
    feats = cfg["features"]
    return [*feats["numeric"], *feats["binary"], *feats["categorical"]]
