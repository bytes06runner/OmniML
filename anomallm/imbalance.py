from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def compute_focal_sample_weights(y_train: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """Focal-inspired per-sample weights for sklearn (Path B), not neural focal loss."""
    labels = np.asarray(y_train, dtype=int)
    counts = np.bincount(labels)
    counts = np.maximum(counts, 1)
    p_class = counts[labels] / float(counts.sum())
    weights = (1.0 / p_class) ** gamma
    return weights * (len(weights) / weights.sum())


def _recommend_strategy(ratio: float, n_minor: int, warnings: list[str]) -> str:
    if ratio >= 0.8:
        return "balanced"
    if ratio >= 0.3:
        return "class_weight"
    if ratio < 0.15:
        if n_minor >= 8:
            return "adasyn"
        if n_minor >= 6:
            return "smote"
        warnings.append("Minority class too small for oversampling; using focal-inspired sample weights.")
        return "focal"
    if n_minor >= 6:
        return "smote"
    warnings.append("Minority class too small for SMOTE; using class weights instead.")
    return "class_weight"


def analyze_class_imbalance(csv_path: str, target_col: Optional[str] = None) -> Dict[str, Any]:
    """Recommend imbalance handling: balanced, class_weight, smote, adasyn, or focal."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    if df.empty:
        return {
            "status": "unknown",
            "ratio": 1.0,
            "n_major": 0,
            "n_minor": 0,
            "recommended_strategy": "balanced",
            "warnings": ["Dataset is empty."],
        }

    resolved_target = target_col or df.columns[-1]
    y_raw = df[resolved_target]
    if y_raw.dtype.kind in ("f",) and y_raw.nunique() > 20:
        return {
            "status": "not_applicable",
            "ratio": 1.0,
            "n_major": len(df),
            "n_minor": len(df),
            "recommended_strategy": "none",
            "warnings": ["Regression task; class imbalance strategies not applied."],
        }

    labels = LabelEncoder().fit_transform(y_raw.astype(str).fillna("missing"))
    counts = pd.Series(labels).value_counts()
    if len(counts) < 2:
        return {
            "status": "single_class",
            "ratio": 1.0,
            "n_major": int(counts.max()) if len(counts) else 0,
            "n_minor": 0,
            "recommended_strategy": "class_weight",
            "warnings": ["Only one class present."],
        }

    n_major = int(counts.max())
    n_minor = int(counts.min())
    ratio = n_minor / n_major if n_major else 1.0
    warnings: list[str] = []
    strategy = _recommend_strategy(ratio, n_minor, warnings)

    return {
        "status": "assessed",
        "ratio": round(ratio, 4),
        "n_major": n_major,
        "n_minor": n_minor,
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "recommended_strategy": strategy,
        "applied_strategy": None,
        "warnings": warnings,
    }
