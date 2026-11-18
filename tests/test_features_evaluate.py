import numpy as np
import pytest

from churn.config import feature_columns
from churn.evaluate import best_threshold, compute_metrics, quality_gate
from churn.features import add_engineered_features, build_pipeline, to_frame


def test_engineered_features(customers):
    out = add_engineered_features(customers)
    row = out.iloc[0]
    assert row["AvgMonthlySpend"] == pytest.approx(row["TotalCharges"] / max(row["tenure"], 1))
    assert 0 <= row["NumServices"] <= 6
    assert set(out["IsNewCustomer"].unique()) <= {0, 1}


@pytest.mark.parametrize("name", ["logistic_regression", "random_forest", "lightgbm"])
def test_pipelines_fit_and_predict(name, customers, base_cfg):
    params = dict(base_cfg["models"][name])
    params.update({"n_estimators": 20} if "n_estimators" in params else {})
    X, y = customers[feature_columns(base_cfg)], customers["Churn"]
    proba = build_pipeline(name, params, base_cfg).fit(X, y).predict_proba(X)[:, 1]
    assert proba.shape == (len(X),)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_to_frame_types():
    df = to_frame([{"tenure": "5", "MonthlyCharges": 10, "TotalCharges": "50.5",
                    "SeniorCitizen": 1}])
    assert df["tenure"].dtype == np.float64 and df["TotalCharges"].iloc[0] == 50.5


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1, 1, 0])
    proba = np.array([0.1, 0.3, 0.7, 0.9, 0.6, 0.4])
    m = compute_metrics(y, proba, threshold=0.5)
    assert m["roc_auc"] == 1.0 and m["recall"] == 1.0
    assert 0.2 <= best_threshold(y, proba) <= 0.8


def test_quality_gate():
    ok, failures = quality_gate({"roc_auc": 0.85, "f1": 0.6}, {"min_roc_auc": 0.8, "min_f1": 0.5})
    assert ok and not failures
    ok, failures = quality_gate({"roc_auc": 0.75}, {"min_roc_auc": 0.8, "min_f1": 0.5})
    assert not ok and len(failures) == 2
