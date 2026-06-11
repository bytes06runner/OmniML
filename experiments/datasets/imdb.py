from __future__ import annotations

from anomallm.featurize import build_imdb_proxy_frame


def load_imdb_frame(sample_size: int = 3000):
    """Text classification proxy: binary 20newsgroups + TF-IDF (offline-friendly)."""
    import pandas as pd

    return build_imdb_proxy_frame(sample_size=sample_size)
