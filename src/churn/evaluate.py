"""Métriques, choix du seuil de décision et quality gate."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, proba, threshold: float = 0.5) -> dict[str, float]:
    pred = (np.asarray(proba) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def best_threshold(y_true, proba) -> float:
    """Seuil qui maximise le F1 sur la validation (le 0.5 par défaut est rarement optimal
    quand les classes sont déséquilibrées)."""
    grid = np.arange(0.20, 0.81, 0.01)
    scores = [f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid]
    return float(round(grid[int(np.argmax(scores))], 2))


def quality_gate(metrics: dict[str, float], gate: dict[str, float]) -> tuple[bool, list[str]]:
    """Retourne (succès, liste des échecs). Chaque clé `min_xxx` est comparée à la métrique `xxx`."""
    failures = []
    for key, minimum in gate.items():
        metric = key.removeprefix("min_")
        value = metrics.get(metric)
        if value is None or value < minimum:
            failures.append(f"{metric} = {value:.3f} < {minimum}" if value is not None
                            else f"{metric} absent")
    return not failures, failures
