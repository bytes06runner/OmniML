from __future__ import annotations

from pathlib import Path


def test_dataset_validation_accepts_txt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = tmp_path / "docs.txt"
    raw.write_text("1\thello world\n0\tbad text\n", encoding="utf-8")

    from graph import dataset_validation_node

    state = {
        "selected_dataset": "local://upload",
        "dataset_download_result": {
            "status": "ok",
            "resolved_path": str(raw),
            "detected_modality": "text",
            "kind": "text_raw",
        },
        "dataset_profile": {},
    }
    result = dataset_validation_node(state)
    assert result["dataset_validation_result"]["status"] == "ok"
    assert result["dataset_validation_result"]["kind"] == "text_raw"
    assert result["modality"] == "text"
