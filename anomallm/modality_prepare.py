from __future__ import annotations

import os
from typing import Any, Dict

from anomallm.config import resolve_training_path
from anomallm.featurize import featurize_image_dir, featurize_text, is_featurized_csv
from anomallm.runtime import ensure_run_paths, load_or_create_bundle, register_artifact


def resolve_modality(state: Dict[str, Any]) -> str:
    task_modality = (state.get("task_representation") or {}).get("modality")
    validation = state.get("dataset_validation_result") or {}
    detected = validation.get("detected_modality")
    if detected in {"text", "image", "tabular"}:
        return detected
    if task_modality in {"text", "image", "tabular"}:
        return task_modality
    csv_path = (state.get("dataset_csv_path") or "").lower()
    if csv_path.endswith((".png", ".jpg", ".jpeg", ".zip")):
        return "image"
    if csv_path.endswith((".txt", ".jsonl")):
        return "text"
    return "tabular"


def run_modality_prepare(state: Dict[str, Any]) -> Dict[str, Any]:
    bundle = load_or_create_bundle(state)
    paths = ensure_run_paths(bundle.run_manifest.run_id)
    modality = resolve_modality(state)
    csv_path = state.get("dataset_csv_path", "")
    validation_kind = (state.get("dataset_validation_result") or {}).get("kind", "")
    features_csv = os.path.join(paths.artifacts, "features.csv")
    featurization_meta: Dict[str, Any] = {}
    use_pytorch = resolve_training_path(state) == "pytorch"
    engineer_template_id = "pytorch_mlp" if use_pytorch else "tabular_sklearn"

    if modality == "tabular" or (csv_path and is_featurized_csv(csv_path)):
        if csv_path and os.path.exists(csv_path):
            features_csv = csv_path
        if not use_pytorch:
            engineer_template_id = "tabular_sklearn"
    elif modality == "text":
        engineer_template_id = "pytorch_mlp" if use_pytorch else "text_sklearn"
        if csv_path and is_featurized_csv(csv_path):
            features_csv = csv_path
        elif csv_path and os.path.exists(csv_path):
            featurization_meta = featurize_text(csv_path, features_csv)
        else:
            raise FileNotFoundError("Text dataset path missing for featurization.")
    elif modality == "image":
        engineer_template_id = "pytorch_mlp" if use_pytorch else "image_sklearn"
        if csv_path and is_featurized_csv(csv_path):
            features_csv = csv_path
        elif csv_path and os.path.exists(csv_path):
            featurization_meta = featurize_image_dir(csv_path, features_csv)
        else:
            raise FileNotFoundError("Image dataset path missing for featurization.")
    else:
        features_csv = csv_path

    if features_csv != csv_path and os.path.exists(features_csv):
        register_artifact(bundle.run_manifest, "features_csv", "dataset", features_csv)

    profile = dict(state.get("dataset_profile") or {})
    profile["modality"] = modality
    profile["csv_path"] = features_csv
    if featurization_meta:
        profile.setdefault("provenance", {})
        if isinstance(profile["provenance"], dict):
            profile["provenance"]["featurization"] = featurization_meta

    return {
        "modality": modality,
        "dataset_csv_path": features_csv,
        "dataset_profile": profile,
        "engineer_template_id": engineer_template_id,
        "featurization_meta": featurization_meta,
        "run_manifest": bundle.run_manifest.model_dump(mode="json"),
    }
