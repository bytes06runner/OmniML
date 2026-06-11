from __future__ import annotations

from anomallm.graph_compile import build_mlp_layers, describe_architecture_layers


def test_build_mlp_layers_dense_dropout_output():
    nodes = [
        {"position": {"y": 0}, "data": {"nodeType": "Dense", "params": {"units": 64, "activation": "relu"}}},
        {"position": {"y": 1}, "data": {"nodeType": "Dropout", "params": {"rate": 0.2}}},
        {"position": {"y": 2}, "data": {"nodeType": "Output", "params": {}}},
    ]
    init_body, forward_body = build_mlp_layers(nodes)
    assert "nn.Linear(input_dim, 64)" in init_body
    assert "nn.Dropout(0.2)" in init_body
    assert "self.output_layer = nn.Linear(64, num_classes)" in init_body
    assert "self.fc0(x)" in forward_body
    assert "self.output_layer(x)" in forward_body


def test_build_mlp_layers_empty_graph_fallback():
    init_body, forward_body = build_mlp_layers([])
    assert "self.fc0 = nn.Linear(input_dim, 128)" in init_body
    assert "self.output_layer = nn.Linear(64, num_classes)" in init_body
    assert "self.fc0(x)" in forward_body


def test_describe_architecture_layers():
    nodes = [
        {"position": {"y": 0}, "data": {"nodeType": "Dense", "params": {"units": 32, "activation": "relu"}}},
    ]
    desc = describe_architecture_layers(nodes)
    assert "Dense(32,relu)" in desc
