from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from experiments.metrics import classification_metrics


def _validation_proba(model: Any, X_val: np.ndarray) -> Optional[np.ndarray]:
    if not hasattr(model, "predict_proba"):
        return None
    proba = model.predict_proba(X_val)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] == 2:
        return proba[:, 1]
    return proba


def run_fold(
    framework: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[Dict[str, float], str]:
    if framework in {"omniml", "sklearn_rf", "rf"}:
        model = RandomForestClassifier(n_estimators=120, random_state=42, class_weight="balanced")
    elif framework in {"sklearn_logreg", "logreg"}:
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:
        raise ValueError(f"Unknown sklearn baseline: {framework}")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return classification_metrics(y_val, preds, _validation_proba(model, X_val)), "ok"


def run_autokeras_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_trials: int = 3,
) -> Tuple[Optional[Dict[str, float]], str]:
    try:
        import autokeras as ak
    except ImportError:
        return None, "skipped_missing_autokeras"

    try:
        clf = ak.StructuredDataClassifier(max_trials=max_trials, overwrite=True)
        clf.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, verbose=0)
        preds = clf.predict(X_val).reshape(-1)
        proba = _validation_proba(clf, X_val) if hasattr(clf, "predict_proba") else None
        return classification_metrics(y_val, preds, proba), "ok"
    except Exception as exc:
        return None, f"failed:{exc}"


def run_h2o_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_runtime_secs: int = 60,
) -> Tuple[Optional[Dict[str, float]], str]:
    try:
        import h2o
        from h2o.automl import H2OAutoML
    except ImportError:
        return None, "skipped_missing_h2o"

    try:
        h2o.init(strict_version_check=False)
        import pandas as pd

        train = pd.DataFrame(X_train, columns=[f"f{i}" for i in range(X_train.shape[1])])
        train["target"] = y_train
        valid = pd.DataFrame(X_val, columns=[f"f{i}" for i in range(X_val.shape[1])])
        valid["target"] = y_val
        train_hf = h2o.H2OFrame(train)
        valid_hf = h2o.H2OFrame(valid)
        aml = H2OAutoML(max_runtime_secs=max_runtime_secs, seed=42, sort_metric="AUC")
        aml.train(y="target", training_frame=train_hf)
        pred = aml.leader.predict(valid_hf).as_data_frame()["predict"].to_numpy()
        return classification_metrics(y_val, pred, None), "ok"
    except Exception as exc:
        return None, f"failed:{exc}"
