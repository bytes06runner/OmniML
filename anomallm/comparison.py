from __future__ import annotations

from typing import Any, Dict, List


def perform_comparative_rag(llm: Any, current_metrics: Dict[str, Any], current_report: str, histories: List[Dict[str, Any]]) -> str:
    if not histories:
        return "This is the pioneer run for this problem; no historical baselines found."

    best_history = max(histories, key=lambda item: float(item.get("val_accuracy") or 0.0))
    current_acc = float(current_metrics.get("val_acc") or current_metrics.get("accuracy") or 0.0)
    best_acc = float(best_history.get("val_accuracy") or 0.0)
    delta = current_acc - best_acc
    direction = "improved" if delta >= 0 else "regressed"
    return (
        f"Compared with the strongest prior run on this problem, the current run {direction} by "
        f"{abs(delta):.4f} on validation accuracy. Prior dataset: {best_history.get('dataset_ref', 'unknown')}."
    )
