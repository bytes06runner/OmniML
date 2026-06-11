from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from anomallm.featurize import (
    build_imdb_proxy_frame,
    featurize_text,
    is_featurized_csv,
    materialize_builtin,
)


def test_build_imdb_proxy_frame():
    frame = build_imdb_proxy_frame(sample_size=50)
    assert "target" in frame.columns
    assert any(str(c).startswith("t") for c in frame.columns)


def test_featurize_text_jsonl(tmp_path):
    raw = tmp_path / "data.jsonl"
    rows = [{"text": "good movie", "label": 1}, {"text": "bad film", "label": 0}]
    with open(raw, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    out_csv = tmp_path / "features.csv"
    meta = featurize_text(str(raw), str(out_csv), max_features=10)
    assert meta["modality"] == "text"
    frame = pd.read_csv(out_csv)
    assert "target" in frame.columns
    assert len(frame) == 2


def test_is_featurized_csv_detects_tfidf(tmp_path):
    path = tmp_path / "f.csv"
    pd.DataFrame({"t0": [1.0], "t1": [0.0], "target": [0]}).to_csv(path, index=False)
    assert is_featurized_csv(str(path)) is True


def test_materialize_builtin_imdb(tmp_path):
    meta = materialize_builtin("omniml/imdb-text-proxy", str(tmp_path))
    assert meta["modality"] == "text"
    assert Path(meta["resolved_path"]).exists()
