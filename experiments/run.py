from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from experiments.baselines.sklearn_baselines import run_autokeras_fold, run_fold, run_h2o_fold
from experiments.datasets.breast_cancer import load_breast_cancer_frame
from experiments.datasets.cifar10 import load_cifar10_frame
from experiments.datasets.imdb import load_imdb_frame
from experiments.metrics import aggregate_fold_metrics


def load_dataset_frame(dataset: str):
    if dataset in {"breast_cancer", "uci_breast_cancer"}:
        return load_breast_cancer_frame()
    if dataset == "imdb":
        return load_imdb_frame()
    if dataset in {"cifar10", "cifar_10"}:
        return load_cifar10_frame()
    raise ValueError(f"Unsupported dataset: {dataset}")


def run_framework_cv(
    framework: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: int,
    seed: int,
) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_metrics: List[Dict[str, float]] = []
    status = "ok"
    for train_idx, val_idx in skf.split(X, y):
        if framework in {"autokeras"}:
            metrics, run_status = run_autokeras_fold(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
            status = run_status
            if metrics is None:
                break
        elif framework in {"h2o", "h2o_automl"}:
            metrics, run_status = run_h2o_fold(X[train_idx], y[train_idx], X[val_idx], y[val_idx])
            status = run_status
            if metrics is None:
                break
        else:
            metrics, run_status = run_fold(framework, X[train_idx], y[train_idx], X[val_idx], y[val_idx])
            status = run_status
        fold_metrics.append(metrics)
    return {
        "status": status,
        "fold_metrics": fold_metrics,
        "summary": aggregate_fold_metrics(fold_metrics) if fold_metrics else {},
    }


def run_experiment(
    dataset: str,
    folds: int,
    seed: int,
    frameworks: List[str],
    ablation: str = "full",
) -> dict:
    frame = load_dataset_frame(dataset)
    X = frame.drop(columns=["target"]).values
    y = frame["target"].values.astype(int)
    if ablation == "monolithic":
        frameworks = ["omniml"]
    results = {
        "dataset": dataset,
        "folds": folds,
        "seed": seed,
        "ablation": ablation,
        "frameworks": {},
    }
    for framework in frameworks:
        results["frameworks"][framework] = run_framework_cv(framework, X, y, folds, seed)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniML paper-aligned offline experiments")
    parser.add_argument("--dataset", default="breast_cancer")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default=str(Path(__file__).parent / "config.json"))
    parser.add_argument(
        "--frameworks",
        default="omniml,sklearn_rf,sklearn_logreg,autokeras,h2o",
        help="Comma-separated framework ids",
    )
    parser.add_argument("--ablation", default="full", choices=["full", "monolithic", "no_xai", "no_hitl"])
    args = parser.parse_args()

    frameworks = [item.strip() for item in args.frameworks.split(",") if item.strip()]
    config_path = Path(args.config)

    def _cli_has(flag: str) -> bool:
        return any(arg == flag or arg.startswith(f"{flag}=") for arg in sys.argv[1:])

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = json.load(handle)
        if not _cli_has("--dataset"):
            args.dataset = cfg.get("dataset", args.dataset)
        if not _cli_has("--folds"):
            args.folds = cfg.get("folds", args.folds)
        if not _cli_has("--seed"):
            args.seed = cfg.get("seed", args.seed)
        if not _cli_has("--frameworks"):
            frameworks = cfg.get("frameworks", frameworks)

    payload = run_experiment(args.dataset, args.folds, args.seed, frameworks, args.ablation)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(__file__).parent / "output" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    lines = [
        "# Experiment Summary",
        "",
        f"- Dataset: `{payload['dataset']}`",
        f"- Folds: `{payload['folds']}`",
        f"- Ablation: `{payload.get('ablation', 'full')}`",
        "",
    ]
    for framework, block in payload["frameworks"].items():
        lines.append(f"## {framework} ({block.get('status', 'unknown')})")
        for metric, stats in (block.get("summary") or {}).items():
            if stats.get("mean") is None:
                continue
            lines.append(
                f"- **{metric}**: {stats['mean']:.4f} ± {stats['std']:.4f} "
                f"(95% CI {stats['ci95'][0]:.4f} – {stats['ci95'][1]:.4f})"
            )
        lines.append("")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {metrics_path}")


if __name__ == "__main__":
    main()
