from __future__ import annotations

from anomallm.featurize import build_cifar_proxy_frame


def load_cifar10_frame(sample_size: int = 5000, random_state: int = 42):
    """CIFAR-10 offline proxy: flattened pixels + sklearn baselines (not UI CNN training)."""
    return build_cifar_proxy_frame(sample_size=sample_size, random_state=random_state)
