from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from .config import get_settings
from .imbalance import analyze_class_imbalance
from .modality_prepare import resolve_modality
from .runtime import ensure_run_paths, load_or_create_bundle, register_artifact, write_json
from .schemas import XAIArtifacts
from .xai import run_xai_for_modality


def modality_node(state: Dict[str, Any]) -> Dict[str, Any]:
    validation = state.get("dataset_validation_result") or {}
    detected = validation.get("detected_modality")
    if detected in {"text", "image", "tabular"}:
        return {"modality": detected}

    task_modality = (state.get("task_representation") or {}).get("modality")
    if task_modality in {"text", "image", "tabular"}:
        return {"modality": task_modality}

    csv_path = state.get("dataset_csv_path", "")
    if csv_path.lower().endswith((".png", ".jpg", ".jpeg", ".zip")):
        return {"modality": "image"}
    if csv_path.lower().endswith((".txt", ".jsonl")):
        return {"modality": "text"}
    return {"modality": resolve_modality(state)}


def imbalance_node(state: Dict[str, Any]) -> Dict[str, Any]:
    csv_path = state.get("dataset_csv_path", "")
    training_config = state.get("training_config") or {}
    if not csv_path or not os.path.exists(csv_path):
        return {"imbalance": {"status": "unknown", "recommended_strategy": "balanced", "warnings": ["Dataset path unavailable."]}}
    target_col = training_config.get("target_column")
    imbalance = analyze_class_imbalance(csv_path, target_col)
    return {"imbalance": imbalance}


def xai_node(state: Dict[str, Any]) -> Dict[str, Any]:
    bundle = load_or_create_bundle(state)
    paths = ensure_run_paths(bundle.run_manifest.run_id)

    if not get_settings().enable_xai:
        return {
            "xai_report": "XAI disabled via OMNIML_ENABLE_XAI=0.",
            "xai_artifacts": XAIArtifacts(status="skipped", explanation_method="disabled").model_dump(mode="json"),
            "run_manifest": bundle.run_manifest.model_dump(mode="json"),
        }

    csv_path = state.get("dataset_csv_path", "")
    training_config = state.get("training_config") or {}
    target_col = training_config.get("target_column")
    export_paths = bundle.training_artifacts.export_paths or {}
    model_path = export_paths.get("model") or os.path.join(paths.exports, "model.pt")

    task_repr = state.get("task_representation") or {}
    modality = state.get("modality") or task_repr.get("modality") or "tabular"
    payload = run_xai_for_modality(
        modality, csv_path, target_col, model_path, paths.plots, paths.artifacts
    )
    model_card = bundle.training_artifacts.model_card
    top_features: List[Dict[str, Any]] = payload.get("top_features") or model_card.get("top_features", [])
    if not top_features and os.path.exists(os.path.join(paths.artifacts, "evaluation.json")):
        try:
            with open(os.path.join(paths.artifacts, "evaluation.json"), "r", encoding="utf-8") as handle:
                evaluation_payload = json.load(handle)
            top_features = evaluation_payload.get("feature_importance", [])[:10]
        except Exception:
            top_features = []

    plot_paths = dict(payload.get("plot_paths") or {})
    feature_plot = os.path.join(paths.plots, "feature_importance.png")
    if os.path.exists(feature_plot):
        plot_paths.setdefault("feature_importance", feature_plot)

    xai = XAIArtifacts(
        status=payload.get("status", "limited"),
        explanation_method=payload.get("explanation_method", "heuristic_fallback"),
        narrative=payload.get("narrative", ""),
        top_features=top_features,
        global_shap=payload.get("global_shap") or {},
        local_lime=payload.get("local_lime") or [],
        plot_paths=plot_paths,
        limitations=payload.get("limitations") or [],
        evidence={"available_plots": list(plot_paths.values())},
    )
    xai_path = os.path.join(paths.artifacts, "xai_summary.json")
    write_json(xai_path, xai.model_dump(mode="json"))
    register_artifact(bundle.run_manifest, "xai_summary", "xai", xai_path)
    if plot_paths.get("local_lime"):
        register_artifact(bundle.run_manifest, "local_lime", "xai", plot_paths["local_lime"])

    training_plots = dict(bundle.training_artifacts.plots or {})
    training_plots.update({k: v for k, v in plot_paths.items() if v})
    bundle.training_artifacts.plots = training_plots

    return {
        "xai_report": xai.narrative,
        "xai_artifacts": xai.model_dump(mode="json"),
        "training_artifacts": bundle.training_artifacts.model_dump(mode="json"),
        "run_manifest": bundle.run_manifest.model_dump(mode="json"),
    }
