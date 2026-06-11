from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Optional, Tuple


def tabular_search_space(task_type: str = "classification") -> Dict[str, Any]:
    """Search space aligned with anomallm/engineer.py grid search."""
    if task_type == "classification":
        candidates = []
        for max_depth, n_estimators in product([4, 8, None], [80, 150]):
            candidates.append({"kind": "rf", "params": {"max_depth": max_depth, "n_estimators": n_estimators}})
        candidates.append({"kind": "logreg", "params": {"C": 1.0}})
        return {
            "task_type": task_type,
            "strategy": "grid",
            "candidates": candidates,
            "param_names": ["kind", "max_depth", "n_estimators", "C"],
        }
    candidates = []
    for max_depth, n_estimators in product([4, 8, None], [80, 150]):
        candidates.append({"kind": "rf_reg", "params": {"max_depth": max_depth, "n_estimators": n_estimators}})
    candidates.append({"kind": "linear", "params": {}})
    return {
        "task_type": task_type,
        "strategy": "grid",
        "candidates": candidates,
        "param_names": ["kind", "max_depth", "n_estimators"],
    }


def pytorch_search_space(
    graph_nodes: List[dict],
    training_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Grid aligned with anomallm/pytorch_engineer.py Path A trials."""
    _ = training_config
    candidates = []
    for lr, batch_size in product([1e-3, 1e-2], [32, 64]):
        candidates.append({"kind": "pytorch", "params": {"learning_rate": float(lr), "batch_size": int(batch_size)}})
    return {
        "task_type": "deep_learning",
        "strategy": "grid",
        "candidates": candidates,
        "param_names": ["learning_rate", "batch_size"],
        "graph_space": deep_learning_search_space(graph_nodes),
        "note": "Path A: PyTorch trains approved architect graph.",
    }


def deep_learning_search_space(graph_nodes: List[dict]) -> Dict[str, Any]:
    """Legacy PyTorch-oriented space from visual graph (Path B design-only)."""
    space: Dict[str, Any] = {
        "learning_rate": ("float_log", 1e-5, 1e-2),
        "batch_size": ("categorical", [16, 32, 64, 128]),
        "optimizer": ("categorical", ["adam", "sgd", "rmsprop"]),
    }
    for node in graph_nodes:
        nid = node.get("id", "")
        ntype = node.get("data", {}).get("nodeType", "")
        if ntype == "Dense":
            space[f"units_{nid}"] = ("categorical", [32, 64, 128, 256, 512])
        elif ntype == "Dropout":
            space[f"rate_{nid}"] = ("float", 0.1, 0.6)
    return {"task_type": "deep_learning", "strategy": "optuna_style", "space": space}


def parse_hpt_from_evaluation(evaluation_payload: Dict[str, Any]) -> Dict[str, Any]:
    trial_results = evaluation_payload.get("trial_results") or []
    if not trial_results:
        return {}
    best = max(trial_results, key=lambda item: float(item.get("value") or 0.0))
    metrics = evaluation_payload.get("metrics") or {}
    return {
        "hpt_best_params": best.get("params") or {},
        "hpt_best_value": float(best.get("value") or 0.0),
        "hpt_trials": trial_results,
        "theta_star": {
            "kind": best.get("kind") or metrics.get("best_kind"),
            "params": best.get("params") or {},
            "value": float(best.get("value") or 0.0),
        },
    }


def parse_hpt_from_stdout_lines(lines: List[str]) -> Dict[str, Any]:
    import json

    trials: List[Dict[str, Any]] = []
    best_params: Dict[str, Any] = {}
    best_value: Optional[float] = None
    best_kind: Optional[str] = None
    for line in lines:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("type") == "hpt_trial":
            trials.append(
                {
                    "trial": payload.get("trial"),
                    "kind": payload.get("params", {}).get("kind") if isinstance(payload.get("params"), dict) else None,
                    "params": payload.get("params"),
                    "value": payload.get("value"),
                }
            )
        elif payload.get("type") == "hpt_complete":
            best_params = payload.get("best_params") or {}
            best_value = float(payload.get("best_value") or 0.0)
            best_kind = payload.get("best_kind")
    if not trials and not best_params:
        return {}
    return {
        "hpt_best_params": best_params,
        "hpt_best_value": best_value,
        "hpt_trials": trials,
        "theta_star": {"kind": best_kind, "params": best_params, "value": best_value},
    }


def merge_approved_params(
    approved: Optional[Dict[str, Any]],
    theta_star: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not approved:
        return None
    kind = approved.get("kind") or (theta_star or {}).get("kind")
    params = approved.get("params") or approved
    if kind and "kind" not in params:
        return {"kind": kind, "params": {k: v for k, v in params.items() if k != "kind"}}
    return approved
