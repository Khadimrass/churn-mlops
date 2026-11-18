"""Simule du trafic de production vers l'API, avec ou sans dérive.

Exemples :
  python scripts/simulate_traffic.py --n 500                # trafic « normal »
  python scripts/simulate_traffic.py --n 500 --drift        # trafic qui a dérivé
  python scripts/simulate_traffic.py --n 500 --url http://localhost:8000

Scénario de dérive simulé : une hausse des prix (+35 % sur MonthlyCharges), un afflux de
nouveaux clients (tenure faible) en contrat mensuel payé par chèque électronique.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.config import feature_columns, load_config  # noqa: E402
from churn.data import load_dataset  # noqa: E402


def apply_drift(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    df = df.copy()
    df["MonthlyCharges"] = (df["MonthlyCharges"] * 1.35).clip(upper=500).round(2)
    df["tenure"] = rng.integers(0, 12, size=len(df))
    df["TotalCharges"] = (df["MonthlyCharges"] * df["tenure"]).round(2)
    mask = rng.random(len(df)) < 0.7
    df.loc[mask, "Contract"] = "Month-to-month"
    df.loc[mask, "PaymentMethod"] = "Electronic check"
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--drift", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config()
    rng = np.random.default_rng(args.seed)
    df = load_dataset(cfg)[feature_columns(cfg)].sample(args.n, replace=True,
                                                        random_state=args.seed)
    if args.drift:
        df = apply_drift(df, rng)

    records = df.to_dict(orient="records")
    n_high = 0
    with httpx.Client(base_url=args.url, timeout=30) as client:
        for i in range(0, len(records), args.batch_size):
            r = client.post("/predict/batch", json={"customers": records[i:i + args.batch_size]})
            r.raise_for_status()
            n_high += r.json()["n_high_risk"]
    mode = "AVEC dérive" if args.drift else "normal"
    print(f"{len(records)} clients envoyés (trafic {mode}) — {n_high} à haut risque")


if __name__ == "__main__":
    main()
