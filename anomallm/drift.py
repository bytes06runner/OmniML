from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def check_feature_drift(reference_csv: str, current_csv: str, threshold: float = 0.2) -> Dict[str, Any]:
    ref = pd.read_csv(reference_csv)
    cur = pd.read_csv(current_csv)
    common = [col for col in ref.columns if col in cur.columns]
    numeric = [col for col in common if pd.api.types.is_numeric_dtype(ref[col]) and pd.api.types.is_numeric_dtype(cur[col])]

    findings: Dict[str, Any] = {}
    for col in numeric:
        ref_col = ref[col].dropna()
        cur_col = cur[col].dropna()
        if ref_col.empty or cur_col.empty:
            continue
        ref_mean = float(ref_col.mean())
        cur_mean = float(cur_col.mean())
        ref_std = float(ref_col.std() or 1.0)
        z_shift = abs(cur_mean - ref_mean) / max(ref_std, 1e-6)
        if z_shift >= threshold:
            findings[col] = {
                "reference_mean": round(ref_mean, 6),
                "current_mean": round(cur_mean, 6),
                "reference_std": round(ref_std, 6),
                "z_shift": round(z_shift, 6),
            }

    if findings:
        return {"status": "drift_detected", "features": findings}
    return {"status": "no_drift", "features": {}}
