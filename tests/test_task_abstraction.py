from __future__ import annotations

from graph import task_abstraction_node


def test_task_abstraction_medical_query():
    state = {
        "user_query": "Diagnose breast cancer from biopsy",
        "run_manifest": {
            "run_id": "test_run",
            "user_query": "Diagnose breast cancer from biopsy",
            "problem_id": "test_run",
            "paths": {},
            "artifact_refs": [],
        },
    }
    result = task_abstraction_node(state)
    assert result["task_representation"]["modality"] == "tabular"
    assert result["task_representation"]["risk_level"] == "high"
