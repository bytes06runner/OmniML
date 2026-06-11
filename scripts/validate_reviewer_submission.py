"""Validate reviewer submission pack consistency."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "experiments" / "output" / "reviewer_submission"
RUN_ID = "breast_cancer_biopsy_c795d4ae"


def main() -> None:
    errors: list[str] = []

    cv_metrics = json.loads((OUT / "cv_results" / "metrics.json").read_text(encoding="utf-8"))
    omniml = cv_metrics["frameworks"]["omniml"]["summary"]["accuracy"]["mean"]
    if abs(omniml - 0.9631) > 0.001:
        errors.append(f"CV accuracy mismatch: {omniml}")

    evaluation = json.loads((OUT / "run_artifacts" / RUN_ID / "evaluation.json").read_text(encoding="utf-8"))
    holdout_acc = evaluation["metrics"]["accuracy"]
    supplementary = (OUT / "SUPPLEMENTARY_MATERIAL.md").read_text(encoding="utf-8")
    if f"{holdout_acc * 100:.1f}%" not in supplementary:
        errors.append("Holdout accuracy not reflected in supplementary doc")

    shots = OUT / "screenshots"
    for name in (
        "training_console_mid.png",
        "training_console_complete.png",
        "final_report_ui.png",
        "final_report_tables.png",
    ):
        if not (shots / name).exists():
            errors.append(f"Missing screenshot: {name}")

    if (shots / "training_console_mid.png").stat().st_size == (shots / "training_console_complete.png").stat().st_size:
        errors.append("Mid and complete console screenshots are identical")

    if not (OUT / "SUPPLEMENTARY_MATERIAL.pdf").exists():
        errors.append("Missing SUPPLEMENTARY_MATERIAL.pdf")

    if errors:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(errors))
    print("Reviewer submission pack validation passed.")


if __name__ == "__main__":
    main()
