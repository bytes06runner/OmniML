from __future__ import annotations

import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from anomallm.xai import run_tabular_xai


def test_run_tabular_xai_generates_artifacts(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame(
        {
            "f1": list(range(40)) + list(range(40, 80)),
            "f2": list(range(80, 0, -1)),
            "target": [0] * 60 + [1] * 20,
        }
    )
    df.to_csv(csv_path, index=False)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(df[["f1", "f2"]], df["target"])
    model_path = tmp_path / "model.pt"
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle)

    plots_dir = tmp_path / "plots"
    artifacts_dir = tmp_path / "artifacts"
    result = run_tabular_xai(str(csv_path), "target", str(model_path), str(plots_dir), str(artifacts_dir))
    assert result["status"] in {"generated", "limited"}
    if result["status"] == "generated":
        assert result.get("top_features") or result.get("local_lime")
