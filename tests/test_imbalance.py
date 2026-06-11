from __future__ import annotations

import pandas as pd

from anomallm.imbalance import analyze_class_imbalance, compute_focal_sample_weights
from anomallm import engineer as engineer_module


def test_analyze_class_imbalance_smote_recommendation(tmp_path):
    path = tmp_path / "imbalanced.csv"
    df = pd.DataFrame({"a": list(range(90)) + list(range(15)), "target": ["A"] * 90 + ["B"] * 15})
    df.to_csv(path, index=False)
    result = analyze_class_imbalance(str(path), "target")
    assert result["status"] == "assessed"
    assert result["recommended_strategy"] == "smote"
    assert 0.15 <= result["ratio"] < 0.3


def test_analyze_class_imbalance_adasyn_recommendation(tmp_path):
    path = tmp_path / "severe.csv"
    df = pd.DataFrame({"a": list(range(90)) + list(range(10)), "target": ["A"] * 90 + ["B"] * 10})
    df.to_csv(path, index=False)
    result = analyze_class_imbalance(str(path), "target")
    assert result["recommended_strategy"] == "adasyn"
    assert result["ratio"] < 0.15
    assert result["n_minor"] >= 8


def test_analyze_class_imbalance_focal_recommendation(tmp_path):
    path = tmp_path / "tiny_minority.csv"
    df = pd.DataFrame({"a": list(range(96)) + list(range(4)), "target": ["A"] * 96 + ["B"] * 4})
    df.to_csv(path, index=False)
    result = analyze_class_imbalance(str(path), "target")
    assert result["recommended_strategy"] == "focal"
    assert result["n_minor"] < 6
    assert any("focal" in w.lower() for w in result["warnings"])


def test_compute_focal_sample_weights_normalized():
    import numpy as np

    y = np.array([0, 0, 0, 1])
    weights = compute_focal_sample_weights(y, gamma=2.0)
    assert len(weights) == 4
    assert abs(weights.mean() - 1.0) < 1e-6
    assert weights[3] > weights[0]


def test_engineer_template_includes_adasyn_and_focal():
    script = engineer_module.engineer_node(
        {
            "run_manifest": {"run_id": "test_run"},
            "dataset_csv_path": "data.csv",
            "imbalance": {"recommended_strategy": "adasyn"},
            "training_config": {},
        }
    )["generated_code"]
    assert "ADASYN" in script
    assert "train_sample_weight" in script
    assert "focal" in script.lower() or 'strategy == "focal"' in script
