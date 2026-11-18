"""Validation des données : on bloque le pipeline si les données ne respectent pas le contrat attendu."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from churn.config import feature_columns

# Valeurs autorisées pour chaque variable catégorielle (contrat de données)
ALLOWED_VALUES: dict[str, set[str]] = {
    "gender": {"Male", "Female"},
    "Partner": {"Yes", "No"},
    "Dependents": {"Yes", "No"},
    "PhoneService": {"Yes", "No"},
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": {"Yes", "No", "No internet service"},
    "OnlineBackup": {"Yes", "No", "No internet service"},
    "DeviceProtection": {"Yes", "No", "No internet service"},
    "TechSupport": {"Yes", "No", "No internet service"},
    "StreamingTV": {"Yes", "No", "No internet service"},
    "StreamingMovies": {"Yes", "No", "No internet service"},
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": {"Yes", "No"},
    "PaymentMethod": {
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    },
}

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "tenure": (0, 120),
    "MonthlyCharges": (0, 500),
    "TotalCharges": (0, 60_000),
    "SeniorCitizen": (0, 1),
}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("Validation des données échouée :\n- " + "\n- ".join(self.errors))


def validate(df: pd.DataFrame, cfg: dict[str, Any], require_target: bool = True) -> ValidationReport:
    report = ValidationReport()
    expected = feature_columns(cfg)
    target = cfg["data"]["target"]

    missing = [c for c in expected if c not in df.columns]
    if require_target and target not in df.columns:
        missing.append(target)
    if missing:
        report.errors.append(f"Colonnes manquantes : {missing}")
        return report

    if len(df) < 100:
        report.errors.append(f"Trop peu de lignes ({len(df)} < 100)")

    for col, allowed in ALLOWED_VALUES.items():
        unknown = set(df[col].dropna().unique()) - allowed
        if unknown:
            report.errors.append(f"{col} : valeurs inattendues {sorted(unknown)}")

    for col, (lo, hi) in NUMERIC_RANGES.items():
        values = pd.to_numeric(df[col], errors="coerce")
        n_nan = int(values.isna().sum())
        if n_nan:
            report.errors.append(f"{col} : {n_nan} valeurs non numériques ou manquantes")
        out = int(((values < lo) | (values > hi)).sum())
        if out:
            report.errors.append(f"{col} : {out} valeurs hors de [{lo}, {hi}]")

    null_rate = df[expected].isna().mean()
    for col, rate in null_rate[null_rate > 0].items():
        report.warnings.append(f"{col} : {rate:.1%} de valeurs manquantes")

    if require_target:
        rate = df[target].mean()
        if not 0.05 <= rate <= 0.60:
            report.errors.append(f"Taux de churn anormal : {rate:.1%}")

    return report
