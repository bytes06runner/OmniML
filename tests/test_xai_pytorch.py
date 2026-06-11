from __future__ import annotations

import json
import os
import tempfile

from anomallm.xai import is_pytorch_export, run_pytorch_limited_xai


def test_is_pytorch_export_from_meta():
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "model.pt")
        with open(model_path, "wb") as handle:
            handle.write(b"not-a-sklearn-pickle")
        meta_path = os.path.join(tmp, "model_meta.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump({"engineer_template_id": "pytorch_mlp", "training_path": "pytorch"}, handle)
        assert is_pytorch_export(model_path) is True


def test_run_pytorch_limited_xai_writes_summary():
    with tempfile.TemporaryDirectory() as tmp:
        plots = os.path.join(tmp, "plots")
        artifacts = os.path.join(tmp, "artifacts")
        os.makedirs(plots)
        os.makedirs(artifacts)
        payload = run_pytorch_limited_xai("tabular", os.path.join(tmp, "model.pt"), artifacts)
        assert payload["explanation_method"] == "pytorch_limited"
        assert os.path.exists(os.path.join(artifacts, "xai_summary.json"))
