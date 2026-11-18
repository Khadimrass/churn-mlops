"""Télécharge le dataset Telco Customer Churn (IBM) dans data/raw/."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churn.config import load_config  # noqa: E402
from churn.data import download_data  # noqa: E402

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-télécharger même si présent")
    args = parser.parse_args()
    path = download_data(load_config(), force=args.force)
    print(f"Données prêtes : {path}")
