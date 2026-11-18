"""Ingestion, nettoyage et découpage des données."""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from churn.config import feature_columns, resolve

logger = logging.getLogger(__name__)


def download_data(cfg: dict[str, Any], force: bool = False) -> Path:
    """Télécharge le CSV brut s'il n'est pas déjà présent."""
    dest = resolve(cfg["data"]["raw_path"])
    if dest.exists() and not force:
        logger.info("Données déjà présentes : %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Téléchargement de %s", cfg["data"]["url"])
    urllib.request.urlretrieve(cfg["data"]["url"], dest)  # noqa: S310 (URL fixe de la config)
    return dest


def clean(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Nettoyage minimal et déterministe — appliqué à l'identique à l'entraînement et en production."""
    df = df.copy()
    # TotalCharges est stocké en texte avec des espaces pour les clients tout neufs (tenure = 0)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    target = cfg["data"]["target"]
    if target in df.columns and not pd.api.types.is_numeric_dtype(df[target]):
        df[target] = (df[target].astype(str).str.strip() == "Yes").astype(int)
    return df


def load_dataset(cfg: dict[str, Any]) -> pd.DataFrame:
    path = resolve(cfg["data"]["raw_path"])
    if not path.exists():
        download_data(cfg)
    df = pd.read_csv(path)
    logger.info("Chargé %d lignes, %d colonnes", *df.shape)
    return clean(df, cfg)


def split_dataset(
    df: pd.DataFrame, cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Découpage stratifié train / validation / test.

    - train : apprentissage
    - val   : sélection du modèle + choix du seuil de décision
    - test  : évaluation finale et quality gate (jamais utilisé pour décider)
    """
    s = cfg["split"]
    target = cfg["data"]["target"]
    X = df[feature_columns(cfg)]
    y = df[target]

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=s["test_size"], stratify=y, random_state=s["random_state"]
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=s["val_size"], stratify=y_tmp, random_state=s["random_state"]
    )
    logger.info("Split — train: %d, val: %d, test: %d", len(X_train), len(X_val), len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test
