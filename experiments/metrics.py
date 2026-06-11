from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def classification_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
        except Exception:
            metrics["auc_roc"] = float("nan")
    else:
        metrics["auc_roc"] = float("nan")
    return metrics


def aggregate_fold_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for key in ("accuracy", "macro_f1", "auc_roc"):
        values = np.array([fold[key] for fold in fold_metrics if key in fold and not np.isnan(fold[key])], dtype=float)
        if len(values) == 0:
            summary[key] = {"mean": None, "std": None, "ci95": [None, None]}
            continue
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci = 1.96 * std / np.sqrt(len(values)) if len(values) > 1 else 0.0
        summary[key] = {"mean": mean, "std": std, "ci95": [mean - ci, mean + ci]}
    return summary
