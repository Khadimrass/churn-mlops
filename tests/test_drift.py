import numpy as np
import pandas as pd

from churn.monitoring.drift import compute_drift, psi_categorical, psi_numeric, render_html


def _with_proba(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["prediction_proba"] = np.random.default_rng(seed).beta(2, 5, len(df))
    return df


def test_psi_is_zero_for_identical_distributions():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi_numeric(x, x) < 1e-6
    s = pd.Series(["a", "b", "b", "c"] * 100)
    assert psi_categorical(s, s) < 1e-6


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert psi_numeric(rng.normal(0, 1, 5000), rng.normal(1.5, 1, 5000)) > 0.2


def test_no_drift_on_same_population(customers, base_cfg):
    ref = _with_proba(customers.iloc[:800], 0)
    cur = _with_proba(customers.iloc[800:], 1)
    report = compute_drift(ref, cur, base_cfg)
    assert report.drift_share == 0
    assert not report.retrain_recommended


def test_drift_detected_on_shifted_population(customers, base_cfg):
    ref = _with_proba(customers.iloc[:800], 0)
    cur = _with_proba(customers.iloc[800:], 1)
    cur["MonthlyCharges"] *= 1.8
    cur["tenure"] = 1
    cur["TotalCharges"] = cur["MonthlyCharges"]
    cur["Contract"] = "Month-to-month"
    cur["PaymentMethod"] = "Electronic check"
    cur["InternetService"] = "Fiber optic"
    cur["prediction_proba"] = np.clip(cur["prediction_proba"] + 0.4, 0, 1)
    report = compute_drift(ref, cur, base_cfg)

    drifted = {f.feature for f in report.features if f.drifted}
    assert {"MonthlyCharges", "tenure", "Contract"} <= drifted
    assert report.prediction_drift and report.retrain_recommended
    assert "Dérive détectée" in render_html(report)
