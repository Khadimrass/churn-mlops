import pandas as pd

from churn.data import clean, split_dataset
from churn.validation import validate


def test_clean_converts_total_charges_and_target(base_cfg):
    raw = pd.DataFrame({
        "TotalCharges": [" ", "29.85"], "tenure": [0, 1],
        "SeniorCitizen": [0, 1], "Churn": ["No", "Yes"],
    })
    out = clean(raw, base_cfg)
    assert out["TotalCharges"].tolist() == [0.0, 29.85]
    assert out["Churn"].tolist() == [0, 1]


def test_valid_data_passes(customers, base_cfg):
    report = validate(customers, base_cfg)
    assert report.ok, report.errors


def test_unknown_category_is_rejected(customers, base_cfg):
    customers.loc[0, "Contract"] = "Lifetime"
    report = validate(customers, base_cfg)
    assert not report.ok
    assert any("Contract" in e for e in report.errors)


def test_missing_column_is_rejected(customers, base_cfg):
    report = validate(customers.drop(columns=["tenure"]), base_cfg)
    assert not report.ok and "manquantes" in report.errors[0]


def test_out_of_range_is_rejected(customers, base_cfg):
    customers.loc[0, "MonthlyCharges"] = -5
    assert not validate(customers, base_cfg).ok


def test_split_is_stratified_and_disjoint(customers, base_cfg):
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(customers, base_cfg)
    assert len(X_train) + len(X_val) + len(X_test) == len(customers)
    assert set(X_train.index).isdisjoint(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.03
