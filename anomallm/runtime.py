from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional

from .schemas import ArtifactRef, DatasetProfile, EvidenceBundle, RunManifest, TrainingArtifacts


@dataclass
class RunPaths:
    root: str
    artifacts: str
    plots: str
    reports: str
    exports: str
    logs: str


def _safe_slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_") or "run"


def ensure_run_paths(run_id: str, cwd: Optional[str] = None) -> RunPaths:
    base = cwd or os.getcwd()
    root = os.path.join(base, "runs", _safe_slug(run_id))
    paths = RunPaths(
        root=root,
        artifacts=os.path.join(root, "artifacts"),
        plots=os.path.join(root, "plots"),
        reports=os.path.join(root, "reports"),
        exports=os.path.join(root, "exports"),
        logs=os.path.join(root, "logs"),
    )
    for path in (paths.root, paths.artifacts, paths.plots, paths.reports, paths.exports, paths.logs):
        os.makedirs(path, exist_ok=True)
    return paths


def write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return path


def write_text(path: str, payload: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)
    return path


def create_run_manifest(run_id: str, user_query: str, problem_id: str, cwd: Optional[str] = None) -> RunManifest:
    paths = ensure_run_paths(run_id, cwd)
    manifest = RunManifest(
        run_id=run_id,
        user_query=user_query,
        problem_id=problem_id,
        paths={
            "root": paths.root,
            "artifacts": paths.artifacts,
            "plots": paths.plots,
            "reports": paths.reports,
            "exports": paths.exports,
            "logs": paths.logs,
        },
    )
    write_json(os.path.join(paths.root, "manifest.json"), manifest.model_dump(mode="json"))
    return manifest


def load_or_create_bundle(state: Dict[str, Any]) -> EvidenceBundle:
    existing = state.get("run_manifest")
    if existing:
        manifest = RunManifest.model_validate(existing)
    else:
        run_id = state.get("run_id") or state.get("problem_id") or f"run_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        manifest = create_run_manifest(run_id, state.get("user_query", ""), state.get("problem_id", run_id))
    from .schemas import XAIArtifacts

    return EvidenceBundle(
        run_manifest=manifest,
        dataset_profile=DatasetProfile.model_validate(state.get("dataset_profile") or {}),
        training_artifacts=TrainingArtifacts.model_validate(state.get("training_artifacts") or {}),
        xai_artifacts=XAIArtifacts.model_validate(state.get("xai_artifacts") or {}),
    )


def register_artifact(manifest: RunManifest, name: str, kind: str, path: str, metadata: Optional[Dict[str, Any]] = None) -> RunManifest:
    manifest.artifact_refs.append(ArtifactRef(name=name, kind=kind, path=path, metadata=metadata or {}))
    manifest.updated_at = datetime.utcnow()
    write_json(os.path.join(manifest.paths["root"], "manifest.json"), manifest.model_dump(mode="json"))
    return manifest


def sync_legacy_export(paths: RunPaths, filename: str) -> None:
    source = os.path.join(paths.exports, filename)
    if not os.path.exists(source):
        return
    legacy_dir = os.path.join(os.getcwd(), "exports")
    os.makedirs(legacy_dir, exist_ok=True)
    shutil.copy2(source, os.path.join(legacy_dir, filename))


def persist_bundle(bundle: EvidenceBundle) -> None:
    manifest = bundle.run_manifest
    write_json(os.path.join(manifest.paths["root"], "manifest.json"), bundle.model_dump(mode="json"))
