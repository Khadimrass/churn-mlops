"""Détection de dérive des données (data drift) et des prédictions (prediction drift).

On compare la distribution des features vues en production (journal de l'API) à celle
du jeu d'entraînement (référence sauvegardée par le pipeline).

  - PSI (Population Stability Index) : mesure l'ampleur du changement
        < 0.10 stable | 0.10-0.20 à surveiller | > 0.20 dérive significative
  - Test de Kolmogorov-Smirnov (numériques) / test du chi² (catégorielles) :
        indique si la différence est statistiquement significative

Une feature est déclarée « en dérive » si PSI > seuil ET p-value < seuil.

Usage : python -m churn.monitoring.drift   (code de sortie 2 si réentraînement recommandé)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from churn.config import load_config, resolve

logger = logging.getLogger(__name__)
EPS = 1e-4


@dataclass
class FeatureDrift:
    feature: str
    kind: str          # numeric | categorical | prediction
    psi: float
    p_value: float
    test: str
    drifted: bool
    reference_stat: str
    current_stat: str


@dataclass
class DriftReport:
    generated_at: str
    n_reference: int
    n_current: int
    drift_share: float
    dataset_drift: bool
    prediction_drift: bool
    retrain_recommended: bool
    features: list[FeatureDrift]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["features"] = [asdict(f) for f in self.features]
        return d


def psi_numeric(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """PSI avec des bacs définis par les quantiles de la référence."""
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # variable quasi constante
        edges = np.unique(np.concatenate([edges, [ref.min() - 1, ref.max() + 1]]))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, edges)[0] / len(ref)
    cur_pct = np.histogram(cur, edges)[0] / len(cur)
    return _psi(ref_pct, cur_pct)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    ref_pct = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    cur_pct = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(ref_pct, cur_pct)


def _psi(ref_pct: np.ndarray, cur_pct: np.ndarray) -> float:
    ref_pct = np.clip(ref_pct, EPS, None)
    cur_pct = np.clip(cur_pct, EPS, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def chi2_pvalue(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    table = np.array([
        ref.value_counts().reindex(cats, fill_value=0).to_numpy(),
        cur.value_counts().reindex(cats, fill_value=0).to_numpy(),
    ])
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return 1.0
    return float(stats.chi2_contingency(table)[1])


def _top_share(s: pd.Series) -> str:
    vc = s.value_counts(normalize=True)
    return f"{vc.index[0]} ({vc.iloc[0]:.0%})"


def compute_drift(reference: pd.DataFrame, current: pd.DataFrame,
                  cfg: dict[str, Any]) -> DriftReport:
    mon = cfg["monitoring"]
    feats = cfg["features"]
    results: list[FeatureDrift] = []

    numeric = [*feats["numeric"], "prediction_proba"]
    for col in numeric:
        ref = pd.to_numeric(reference[col], errors="coerce").dropna().to_numpy()
        cur = pd.to_numeric(current[col], errors="coerce").dropna().to_numpy()
        psi = psi_numeric(ref, cur)
        p = float(stats.ks_2samp(ref, cur).pvalue)
        results.append(FeatureDrift(
            feature=col, kind="prediction" if col == "prediction_proba" else "numeric",
            psi=round(psi, 4), p_value=p, test="KS",
            drifted=bool(psi > mon["psi_threshold"] and p < mon["p_value_threshold"]),
            reference_stat=f"moy. {ref.mean():.2f}", current_stat=f"moy. {cur.mean():.2f}",
        ))

    for col in [*feats["binary"], *feats["categorical"]]:
        ref, cur = reference[col].astype(str), current[col].astype(str)
        psi = psi_categorical(ref, cur)
        p = chi2_pvalue(ref, cur)
        results.append(FeatureDrift(
            feature=col, kind="categorical", psi=round(psi, 4), p_value=p, test="chi²",
            drifted=bool(psi > mon["psi_threshold"] and p < mon["p_value_threshold"]),
            reference_stat=_top_share(ref), current_stat=_top_share(cur),
        ))

    input_feats = [r for r in results if r.kind != "prediction"]
    share = sum(r.drifted for r in input_feats) / len(input_feats)
    pred = next(r for r in results if r.kind == "prediction")
    results.sort(key=lambda r: r.psi, reverse=True)
    return DriftReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        n_reference=len(reference), n_current=len(current),
        drift_share=round(share, 4),
        dataset_drift=share > mon["drift_share_threshold"],
        prediction_drift=pred.drifted,
        # Alerte si beaucoup de features dérivent OU si le comportement du modèle change
        retrain_recommended=share > mon["drift_share_threshold"] or pred.drifted,
        features=results,
    )


def load_production_log(path: str | Path, last_n: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_json(path, lines=True)
    return df.tail(last_n) if last_n else df


def render_html(report: DriftReport) -> str:
    status = ("Dérive détectée : réentraînement recommandé" if report.retrain_recommended
              else "Pas de dérive significative")
    rows = "\n".join(
        f"<tr class='{'bad' if f.drifted else ''}'><td>{f.feature}</td><td>{f.kind}</td>"
        f"<td>{f.psi:.3f}</td><td>{f.test}</td><td>{f.p_value:.2e}</td>"
        f"<td>{f.reference_stat}</td><td>{f.current_stat}</td>"
        f"<td>{'oui' if f.drifted else 'non'}</td></tr>"
        for f in report.features
    )
    return f"""<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<title>Rapport de dérive (Churn)</title><style>
body{{font-family:system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1f2933}}
h1{{margin-bottom:.2rem}} .muted{{color:#616e7c}}
.cards{{display:flex;gap:1rem;margin:1.5rem 0}}
.card{{flex:1;border:1px solid #e4e7eb;border-radius:10px;padding:1rem}}
.card b{{font-size:1.6rem;display:block}}
table{{width:100%;border-collapse:collapse;font-size:.92rem}}
th,td{{padding:.5rem;border-bottom:1px solid #e4e7eb;text-align:left}}
th{{background:#f5f7fa}} tr.bad{{background:#fff1f0}}
</style></head><body>
<h1>Rapport de dérive des données</h1>
<p class="muted">Généré le {report.generated_at}. Référence : {report.n_reference} lignes
(entraînement), production : {report.n_current} lignes</p>
<h2>{status}</h2>
<div class="cards">
 <div class="card"><span class="muted">Features en dérive</span><b>{report.drift_share:.0%}</b></div>
 <div class="card"><span class="muted">Dérive des prédictions</span>
  <b>{'Oui' if report.prediction_drift else 'Non'}</b></div>
 <div class="card"><span class="muted">Échantillon analysé</span><b>{report.n_current}</b></div>
</div>
<table><thead><tr><th>Feature</th><th>Type</th><th>PSI</th><th>Test</th><th>p-value</th>
<th>Référence</th><th>Production</th><th>Dérive</th></tr></thead><tbody>
{rows}
</tbody></table>
<p class="muted">PSI &lt; 0.10 : stable, de 0.10 à 0.20 : à surveiller, &gt; 0.20 : dérive.
Une feature est en dérive si PSI &gt; seuil et p-value &lt; seuil.</p>
</body></html>"""


def run(config_path: str | None = None, last_n: int | None = None) -> DriftReport | None:
    cfg = load_config(config_path)
    mon = cfg["monitoring"]
    reference = pd.read_parquet(resolve(cfg["artifacts"]["reference_path"]))
    current = load_production_log(resolve(mon["predictions_log"]), last_n)
    if len(current) < mon["min_samples"]:
        logger.warning("Pas assez de prédictions en production (%d < %d)",
                       len(current), mon["min_samples"])
        return None

    report = compute_drift(reference, current, cfg)
    out = resolve(mon["reports_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "drift_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "drift_report.html").write_text(render_html(report), encoding="utf-8")
    logger.info("Rapport écrit dans %s (drift_share = %.0f%%, réentraînement = %s)",
                out, report.drift_share * 100, report.retrain_recommended)
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="Analyse la dérive des données de production.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--last-n", type=int, default=None,
                        help="N'analyser que les N dernières prédictions")
    args = parser.parse_args()
    report = run(args.config, args.last_n)
    if report is None:
        sys.exit(0)
    for f in report.features[:5]:
        logger.info("  %-18s PSI=%.3f  dérive=%s", f.feature, f.psi, f.drifted)
    sys.exit(2 if report.retrain_recommended else 0)


if __name__ == "__main__":
    main()
