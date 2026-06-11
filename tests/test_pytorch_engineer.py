from __future__ import annotations

import ast
import os

import pytest

from anomallm.config import resolve_training_path
from anomallm.graph_compile import build_mlp_layers
from anomallm.hpo import pytorch_search_space
from anomallm.pytorch_engineer import generate_pytorch_training_script


def test_resolve_training_path_env_pytorch(monkeypatch):
    monkeypatch.delenv("OMNIML_TRAINING_PATH", raising=False)
    monkeypatch.setenv("OMNIML_TRAINING_PATH", "pytorch")
    assert resolve_training_path({}) == "pytorch"


def test_resolve_training_path_state_overrides_env(monkeypatch):
    monkeypatch.setenv("OMNIML_TRAINING_PATH", "pytorch")
    state = {"training_config": {"training_path": "sklearn"}}
    assert resolve_training_path(state) == "sklearn"


def test_resolve_training_path_state_pytorch(monkeypatch):
    monkeypatch.setenv("OMNIML_TRAINING_PATH", "sklearn")
    state = {"training_config": {"training_path": "pytorch"}}
    assert resolve_training_path(state) == "pytorch"


def test_pytorch_search_space_candidates():
    space = pytorch_search_space([], {})
    kinds = {c.get("kind") for c in space.get("candidates", [])}
    assert kinds == {"pytorch"}
    assert len(space["candidates"]) == 4


def test_generate_pytorch_training_script_contract():
    nodes = [
        {"position": {"y": 0}, "data": {"nodeType": "Dense", "params": {"units": 16, "activation": "relu"}}},
        {"position": {"y": 1}, "data": {"nodeType": "Output", "params": {}}},
    ]
    init_body, _ = build_mlp_layers(nodes)
    state = {
        "run_manifest": {"run_id": "test_pytorch_run"},
        "dataset_csv_path": "data.csv",
        "user_query": "test",
        "training_config": {"epochs": 2, "batch_size": 8, "lr": 0.01, "optimizer": "adam"},
        "graph_architecture_json": {"nodes": nodes},
        "imbalance": {},
        "modality": "tabular",
    }
    script = generate_pytorch_training_script(state)
    ast.parse(script)
    assert "OmniMLNet" in script
    assert '"type": "epoch_metric"' in script or '"type":"epoch_metric"' in script
    assert init_body.strip() in script
    assert "pytorch_mlp" in script
    assert "torch.save(final_model.state_dict()" in script


def test_engineer_node_routes_pytorch(monkeypatch):
    monkeypatch.setenv("OMNIML_TRAINING_PATH", "pytorch")
    from anomallm.engineer import engineer_node

    state = {
        "run_manifest": {"run_id": "route_test"},
        "dataset_csv_path": "x.csv",
        "user_query": "q",
        "training_config": {},
        "graph_architecture_json": {"nodes": []},
        "imbalance": {},
    }
    out = engineer_node(state)
    assert out.get("engineer_template_id") == "pytorch_mlp"
    assert "OmniMLNet" in out.get("generated_code", "")
