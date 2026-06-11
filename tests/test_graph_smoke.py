from __future__ import annotations

import os


def test_build_graph_compiles():
    os.environ.setdefault("OMNIML_SKIP_KAGGLE_AUTH", "1")
    os.environ.setdefault("GROQ_API_KEY", "test-key")

    from graph import build_graph

    compiled = build_graph()
    assert compiled is not None
    node_names = set(compiled.get_graph().nodes.keys())
    assert "architect" in node_names
    assert "xai_node" in node_names
    assert "compliance_renderer" in node_names
