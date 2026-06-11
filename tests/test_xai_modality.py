from __future__ import annotations

import pickle

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from anomallm.xai import run_image_xai, run_text_xai, run_xai_for_modality


def _tiny_text_run(tmp_path):
    csv_path = tmp_path / "text.csv"
    df = pd.DataFrame(
        {
            "t0": [1.0, 0.0, 1.0, 0.0],
            "t1": [0.0, 1.0, 0.0, 1.0],
            "target": [0, 1, 0, 1],
        }
    )
    df.to_csv(csv_path, index=False)
    model = LogisticRegression(max_iter=500)
    model.fit(df[["t0", "t1"]], df["target"])
    model_path = tmp_path / "model.pt"
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)
    plots_dir = tmp_path / "plots"
    artifacts_dir = tmp_path / "artifacts"
    return str(csv_path), "target", str(model_path), str(plots_dir), str(artifacts_dir)


def test_run_xai_for_modality_text(tmp_path):
    csv_path, target, model_path, plots_dir, artifacts_dir = _tiny_text_run(tmp_path)
    result = run_xai_for_modality("text", csv_path, target, model_path, plots_dir, artifacts_dir)
    assert "text" in result.get("explanation_method", "")
    assert result.get("modality") == "text"


def test_run_text_xai_limited_without_csv(tmp_path):
    result = run_text_xai("", None, str(tmp_path / "m.pt"), str(tmp_path / "p"), str(tmp_path / "a"))
    assert result["status"] == "limited"
    assert "text" in result["explanation_method"]


def test_run_image_xai_limited_without_sklearn(tmp_path):
    torch_path = tmp_path / "model.pt"
    torch_path.write_bytes(b"not-a-model")
    result = run_image_xai(
        str(tmp_path / "missing.csv"),
        None,
        str(torch_path),
        str(tmp_path / "plots"),
        str(tmp_path / "artifacts"),
    )
    assert result["status"] == "limited"
    assert "image" in result["explanation_method"]
