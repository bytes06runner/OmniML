from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anomallm.runtime import create_run_manifest, ensure_run_paths
from anomallm.schemas import DatasetProfile, EvidenceBundle, TrainingArtifacts, XAIArtifacts


@pytest.fixture(autouse=True)
def _skip_kaggle_auth(monkeypatch):
    monkeypatch.setenv("OMNIML_SKIP_KAGGLE_AUTH", "1")


@pytest.fixture
def sample_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manifest = create_run_manifest("test_run", "Diagnose breast cancer", "problem_test")
    paths = ensure_run_paths("test_run")
    bundle = EvidenceBundle(
        run_manifest=manifest,
        dataset_profile=DatasetProfile(
            csv_path=str(tmp_path / "data.csv"),
            dataset_ref="test/dataset",
            row_count=100,
            provenance={"summary": "Synthetic fixture"},
        ),
        training_artifacts=TrainingArtifacts(
            metrics={"accuracy": 0.9, "val_acc": 0.9},
            model_card={"top_features": [{"feature": "f1", "importance": 0.5}]},
        ),
        xai_artifacts=XAIArtifacts(
            status="generated",
            explanation_method="shap_global+lime_local",
            narrative="Test narrative",
            top_features=[{"feature": "f1", "importance": 0.5}],
        ),
    )
    return bundle
