from __future__ import annotations

from anomallm.hpo import parse_hpt_from_evaluation, tabular_search_space


def test_tabular_search_space_has_candidates():
    space = tabular_search_space("classification")
    assert len(space["candidates"]) >= 3


def test_parse_hpt_from_evaluation():
    payload = {
        "trial_results": [
            {"trial": 1, "kind": "rf", "params": {"max_depth": 4}, "value": 0.9},
            {"trial": 2, "kind": "logreg", "params": {"C": 1.0}, "value": 0.8},
        ]
    }
    parsed = parse_hpt_from_evaluation(payload)
    assert parsed["hpt_best_params"] == {"max_depth": 4}
    assert parsed["theta_star"]["kind"] == "rf"
