"""Construction des pipelines scikit-learn (préprocessing + modèle dans un seul objet)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Features métier simples, calculées dans le pipeline pour éviter tout décalage train/serving."""
    X = X.copy()
    X["AvgMonthlySpend"] = X["TotalCharges"] / X["tenure"].clip(lower=1)
    services = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    X["NumServices"] = (X[services] == "Yes").sum(axis=1)
    X["IsNewCustomer"] = (X["tenure"] <= 6).astype(int)
    return X


ENGINEERED_NUMERIC = ["AvgMonthlySpend", "NumServices", "IsNewCustomer"]


def build_preprocessor(cfg: dict[str, Any]) -> ColumnTransformer:
    feats = cfg["features"]
    numeric = [*feats["numeric"], *feats["binary"], *ENGINEERED_NUMERIC]
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", categorical_pipe, feats["categorical"])],
        verbose_feature_names_out=False,
    )


def build_estimator(name: str, params: dict[str, Any], random_state: int):
    if name == "logistic_regression":
        return LogisticRegression(**params, random_state=random_state)
    if name == "random_forest":
        return RandomForestClassifier(**params, random_state=random_state)
    if name == "lightgbm":
        return LGBMClassifier(**params, random_state=random_state)
    raise ValueError(f"Modèle inconnu : {name}")


def build_pipeline(name: str, params: dict[str, Any], cfg: dict[str, Any]) -> Pipeline:
    return Pipeline(
        [
            ("engineer", FunctionTransformer(add_engineered_features, validate=False)),
            ("preprocess", build_preprocessor(cfg)),
            ("model", build_estimator(name, params, cfg["split"]["random_state"])),
        ]
    )


def to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convertit des enregistrements JSON (API) en DataFrame typé comme à l'entraînement."""
    df = pd.DataFrame.from_records(records)
    for col in ("tenure", "MonthlyCharges", "TotalCharges"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
    df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)
    return df
