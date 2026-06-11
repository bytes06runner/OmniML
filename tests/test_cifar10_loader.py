from __future__ import annotations

import pytest


def test_load_cifar10_frame_shape():
    try:
        from experiments.datasets.cifar10 import load_cifar10_frame
    except ImportError:
        pytest.skip("torchvision not installed")

    frame = load_cifar10_frame(sample_size=32, random_state=0)
    assert "target" in frame.columns
    assert len(frame) == 32
    assert frame["target"].nunique() >= 1
    feature_cols = [c for c in frame.columns if c != "target"]
    assert len(feature_cols) == 3072
    assert feature_cols[0] == "p0"
