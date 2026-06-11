from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer


def load_breast_cancer_frame() -> pd.DataFrame:
    data = load_breast_cancer()
    frame = pd.DataFrame(data.data, columns=data.feature_names)
    frame["target"] = data.target
    return frame
