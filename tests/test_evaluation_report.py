from __future__ import annotations

import json
from pathlib import Path

from anomallm.evaluation_report import render_evaluation_sections, search_space_table_rows
from anomallm.hpo import tabular_search_space


def test_search_space_table_uses_actual_tabular_values():
    space = tabular_search_space("classification")
    rows = dict(search_space_table_rows(space, {"epochs": 5, "optimizer": "adam", "batch_size": 64}))
    assert rows["Model / Estimator"] == "logreg, rf"
    assert rows["Max Depth"] == "4, 8, unlimited"
    assert rows["N Estimators"] == "80, 150"
    assert rows["Regularization (C)"] == "1.0"


def test_render_evaluation_sections_includes_protocol_and_table():
    space = tabular_search_space("classification")
    markdown = render_evaluation_sections(
        metrics={"accuracy": 0.965, "f1": 0.964},
        architecture="Dense(128,relu) → Output(1,sigmoid)",
        training_config={"epochs": 5, "optimizer": "adam"},
        hpt_search_space=space,
        epoch_metrics=[{"epoch": 5}],
        best_params={"max_depth": 8, "n_estimators": 150},
    )
    assert "## Evaluation Protocol" in markdown
    assert "96.5%" in markdown
    assert "0.964" in markdown
    assert "## Hyperparameter Search Space" in markdown
    assert "| Max Depth | 4, 8, unlimited |" in markdown


def test_render_from_existing_run_fixture():
    repo_root = Path(__file__).resolve().parents[1]
    evaluation_path = repo_root / "runs" / "breast_cancer_biopsy_c795d4ae" / "artifacts" / "evaluation.json"
    if not evaluation_path.exists():
        return
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    markdown = render_evaluation_sections(
        metrics=payload["metrics"],
        architecture="Dense(128,relu) → BN1d → Dropout(0.3) → Dense(64,relu)",
        training_config={"epochs": 5, "optimizer": "adam"},
        hpt_search_space=tabular_search_space("classification"),
        best_params={"max_depth": 4, "n_estimators": 80},
    )
    assert "Evaluation Protocol" in markdown
    assert "Hyperparameter Search Space" in markdown
