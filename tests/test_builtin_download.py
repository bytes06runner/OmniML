from __future__ import annotations

from pathlib import Path

from anomallm.builtin_datasets import builtin_download_tool, local_upload_download_tool


def test_builtin_download_imdb_proxy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = builtin_download_tool("omniml/imdb-text-proxy", run_id="test_run")
    assert result["status"] == "ok"
    assert result["source"] == "builtin"
    assert Path(result["resolved_path"]).exists()


def test_local_upload_download_txt(tmp_path):
    raw = tmp_path / "sample.txt"
    raw.write_text("1\tpositive review\n0\tnegative review\n", encoding="utf-8")
    ref = f"local://{raw.as_posix()}"
    result = local_upload_download_tool(ref)
    assert result["status"] == "ok"
    assert result["detected_modality"] == "text"
    assert result["kind"] == "text_raw"
