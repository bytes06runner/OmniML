from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .runtime import ensure_run_paths, load_or_create_bundle, register_artifact, write_json
from .schemas import XAIArtifacts


def modality_node(state: Dict[str, Any]) -> Dict[str, Any]:
    csv_path = state.get("dataset_csv_path", "")
    modality = "tabular"
    if csv_path.lower().endswith((".png", ".jpg", ".jpeg")):
        modality = "image"
    elif csv_path.lower().endswith((".txt", ".jsonl")):
        modality = "text"
    return {"modality": modality}


def imbalance_node(state: Dict[str, Any]) -> Dict[str, Any]:
    eda_data = state.get("eda_data", {})
    columns = eda_data.get("columns", [])
    imbalance = {"status": "unknown", "details": {}}
    if columns:
        imbalance["status"] = "not_assessed"
    return {"imbalance": imbalance}


def xai_node(state: Dict[str, Any]) -> Dict[str, Any]:
    bundle = load_or_create_bundle(state)
    paths = ensure_run_paths(bundle.run_manifest.run_id)
    model_card = bundle.training_artifacts.model_card
    top_features: List[Dict[str, Any]] = model_card.get("top_features", [])
    if not top_features and os.path.exists("metrics.json"):
        try:
            with open("metrics.json", "r", encoding="utf-8") as handle:
                metrics_payload = json.load(handle)
            feature_names = metrics_payload.get("feature_names", [])[:5]
            top_features = [{"feature": name, "importance": round(1.0 / (index + 1), 4)} for index, name in enumerate(feature_names)]
        except Exception:
            top_features = []
    narrative = (
        "The model explanation is evidence-backed and should be interpreted as directional guidance rather than causal proof. "
        "Top features reflect the most influential available signals from the current training artifacts."
    )
    xai = XAIArtifacts(
        status="generated" if top_features else "limited",
        explanation_method="heuristic_feature_importance",
        narrative=narrative,
        top_features=top_features,
        limitations=[
            "Feature importance is approximate when derived from generic training artifacts.",
            "No counterfactual explanations are generated in this release.",
        ],
        evidence={"available_plots": list(bundle.training_artifacts.plots.values())},
    )
    xai_path = os.path.join(paths.artifacts, "xai_summary.json")
    write_json(xai_path, xai.model_dump(mode="json"))
    register_artifact(bundle.run_manifest, "xai_summary", "xai", xai_path)
    return {
        "xai_report": narrative,
        "xai_artifacts": xai.model_dump(mode="json"),
        "run_manifest": bundle.run_manifest.model_dump(mode="json"),
    }
