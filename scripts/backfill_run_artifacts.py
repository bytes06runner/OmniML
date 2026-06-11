#!/usr/bin/env python3
"""Backfill hpt_summary.json, final_report.md, and evidence_appendix.md for a run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reviewer_utils import (  # noqa: E402
    DEFAULT_RUN_ID,
    backfill_hpt_summary,
    run_root,
    write_final_report,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args()

    hpt_path = backfill_hpt_summary(args.run_id)
    report_path = write_final_report(args.run_id)
    manifest = run_root(args.run_id) / "manifest.json"
    appendix = run_root(args.run_id) / "evidence_appendix.md"
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "export_evidence_appendix.py"), str(manifest), "-o", str(appendix)],
        check=True,
    )
    print(f"Wrote {hpt_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote {appendix}")


if __name__ == "__main__":
    main()
