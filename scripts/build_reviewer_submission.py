#!/usr/bin/env python3
"""Build authentic reviewer submission pack (markdown, PDF, zip, screenshots)."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_reviewer_utils import (  # noqa: E402
    DEFAULT_RUN_ID,
    build_hpt_search_space,
    latest_cv_output_dir,
    load_evaluation,
    load_manifest_bundle,
    render_report_html_for_run,
    run_root,
    write_supplementary_pdf,
)

OUT = ROOT / "experiments" / "output" / "reviewer_submission"


def _read_cv_summary(cv_dir: Path) -> str:
    return (cv_dir / "summary.md").read_text(encoding="utf-8")


def build_supplementary_markdown(run_id: str, cv_dir: Path) -> str:
    from anomallm.evaluation_report import render_search_space_table

    evaluation = load_evaluation(run_id)
    metrics = evaluation.get("metrics") or {}
    hpt = (run_root(run_id) / "artifacts" / "hpt_summary.json")
    best_params = {}
    if hpt.exists():
        best_params = json.loads(hpt.read_text(encoding="utf-8")).get("hpt_best_params") or {}

    cv_summary = _read_cv_summary(cv_dir)
    search_table = render_search_space_table(build_hpt_search_space(), {"epochs": 5, "optimizer": "adam"})

    return f"""# OmniML Supplementary Material

## 1. Evaluation Protocol

### 1.1 Paper metrics (5-fold cross-validation)

Primary Table II numbers are produced offline by `experiments/run.py` on the **sklearn UCI breast cancer** dataset (569 samples, 30 features, binary target).

{cv_summary}

**Reproduce:**

```bash
pip install -r requirements.txt
python -m experiments.run --dataset breast_cancer --folds 5 --seed 42 --frameworks omniml,sklearn_rf,sklearn_logreg
python -m experiments.compare --latest
```

### 1.2 Chainlit UI audit run (holdout evaluation)

Run ID: `{run_id}`

| Field | Value |
| --- | --- |
| Protocol | Single 80/20 stratified holdout on Kaggle Wisconsin diagnostic CSV |
| Accuracy | {float(metrics.get('accuracy', metrics.get('val_acc', 0))) * 100:.1f}% |
| F1 | {float(metrics.get('f1', 0)):.3f} |
| Epochs | 5 configured (console epoch curves are telemetry for UI; Path B uses sklearn) |
| Optimizer (config) | Adam (UI config); actual search uses sklearn RF/LogReg grid |
| Selected estimator | RandomForest |
| Selected hyperparameters | `{best_params}` |

### 1.3 Architecture vs training path (transparency note)

- **HITL-approved graph:** Dense(128,relu) → BN1d → Dropout → Dense layers → Output (design artifact)
- **Default execution (Path B):** sklearn grid search on featurized CSV — this is what produced both CV benchmarks and the cited UI run estimator
- **Optional Path A:** Set `training_path=pytorch` to compile the approved graph to PyTorch

## 2. Hyperparameter Search Space

{search_table}

## 3. Screenshot captions

| File | Description |
| --- | --- |
| `screenshots/training_console_mid.png` | Live Training Console replayed from `runs/{run_id}/logs/training.log` (mid-run epoch curves) |
| `screenshots/training_console_complete.png` | Same replay at completion showing Evaluation Protocol + Hyperparameter Search Space panels |
| `screenshots/final_report_ui.png` | Final evaluation report rendered from run artifacts |
| `screenshots/final_report_tables.png` | Evaluation Protocol and HPT tables (full contrast, light theme) |

## 4. Artifact index

| Artifact | Path |
| --- | --- |
| CV metrics JSON | `cv_results/metrics.json` |
| CV summary | `cv_results/summary.md` |
| Table II | `cv_results/table_ii.md` |
| Run evaluation | `run_artifacts/{run_id}/evaluation.json` |
| HPT summary | `run_artifacts/{run_id}/hpt_summary.json` |
| Final report | `run_artifacts/{run_id}/final_report.md` |
| Evidence appendix | `run_artifacts/{run_id}/evidence_appendix.md` |
"""


def build_cover_letter(run_id: str, cv_dir: Path) -> str:
    cv_summary = _read_cv_summary(cv_dir)
    omniml_block = ""
    for line in cv_summary.splitlines():
        if line.startswith("- **accuracy**"):
            omniml_block += line + "\n"
        if line.startswith("- **macro_f1**"):
            omniml_block += line + "\n"
        if line.startswith("- **auc_roc**"):
            omniml_block += line + "\n"

    return f"""# Cover Letter (email template)

Dear Editor and Reviewers,

Thank you for accepting our paper. As requested, we attach supplementary material documenting the **evaluation protocol** and **hyperparameter search space** used in our experiments.

**Primary reproducible results (5-fold CV, UCI breast cancer):**

{omniml_block}

**Hyperparameter search:** 7-point grid over RandomForest (`max_depth` ∈ {{4, 8, unlimited}}, `n_estimators` ∈ {{80, 150}}) and LogisticRegression (`C` = 1.0), implemented in `anomallm/hpo.py`.

**Attached files:**
1. `SUPPLEMENTARY_MATERIAL.pdf` — full protocol, search space table, reproducibility commands
2. `reviewer_submission.zip` — CV metrics, run artifacts for `{run_id}`, and Training Console screenshots replayed from real `training.log` JSON telemetry

**Important distinction:** Paper Table II numbers come from offline CV (`experiments/run.py`). Console screenshots document the HITL transparency UI (epochs, optimizer config, architecture, search space) from run `{run_id}` on a single holdout split.

Best regards,
[Authors]
"""


def copy_tree(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_zip(out_dir: Path) -> Path:
    zip_path = out_dir / "reviewer_submission.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in out_dir.rglob("*"):
            if path == zip_path or path.suffix == ".zip":
                continue
            if path.is_file():
                zf.write(path, arcname=path.relative_to(out_dir).as_posix())
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--skip-screenshots", action="store_true")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    subprocess.run([sys.executable, str(ROOT / "scripts" / "backfill_run_artifacts.py"), "--run-id", args.run_id], check=True)

    cv_dir = latest_cv_output_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cv_results").mkdir(exist_ok=True)
    shutil.copy2(cv_dir / "metrics.json", OUT / "cv_results" / "metrics.json")
    shutil.copy2(cv_dir / "summary.md", OUT / "cv_results" / "summary.md")
    shutil.copy2(ROOT / "experiments" / "output" / "table_ii.md", OUT / "cv_results" / "table_ii.md")

    run_out = OUT / "run_artifacts" / args.run_id
    run_out.mkdir(parents=True, exist_ok=True)
    artifact_map = {
        "evaluation.json": run_root(args.run_id) / "artifacts" / "evaluation.json",
        "hpt_summary.json": run_root(args.run_id) / "artifacts" / "hpt_summary.json",
        "final_report.md": run_root(args.run_id) / "reports" / "final_report.md",
        "evidence_appendix.md": run_root(args.run_id) / "evidence_appendix.md",
    }
    for name, src in artifact_map.items():
        if src.exists():
            shutil.copy2(src, run_out / name)

    supplementary = build_supplementary_markdown(args.run_id, cv_dir)
    (OUT / "SUPPLEMENTARY_MATERIAL.md").write_text(supplementary, encoding="utf-8")
    (OUT / "COVER_LETTER.md").write_text(build_cover_letter(args.run_id, cv_dir), encoding="utf-8")

    from anomallm.evaluation_report import render_search_space_table

    (OUT / "evaluation_protocol.md").write_text(
        supplementary.split("## 2. Hyperparameter Search Space")[0].strip() + "\n",
        encoding="utf-8",
    )
    (OUT / "hyperparameter_search_space.md").write_text(
        render_search_space_table(build_hpt_search_space(), {"epochs": 5, "optimizer": "adam"}) + "\n",
        encoding="utf-8",
    )

    write_supplementary_pdf(OUT / "SUPPLEMENTARY_MATERIAL.md", OUT / "SUPPLEMENTARY_MATERIAL.pdf")
    (OUT / "final_report_preview.html").write_text(render_report_html_for_run(args.run_id), encoding="utf-8")

    readme = f"""# Reviewer Submission Pack

Authentic artifacts for post-acceptance reviewer requests.

## Contents

- `SUPPLEMENTARY_MATERIAL.pdf` — send with email
- `COVER_LETTER.md` — email body template
- `cv_results/` — 5-fold CV metrics ({cv_dir.name})
- `run_artifacts/{args.run_id}/` — Chainlit run audit trail
- `screenshots/` — Training Console replay from real training.log

## Reproduce CV

```bash
python -m experiments.run --dataset breast_cancer --folds 5 --seed 42
python -m experiments.compare --latest
```

## Reproduce console screenshots

```bash
python scripts/reviewer_console_server.py --run-id {args.run_id}
python scripts/capture_reviewer_screenshots.py
```

> Supersedes the staged demo pack in `experiments/output/reviewer_share/`.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")

    if not args.skip_screenshots:
        mid_server = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts" / "reviewer_console_server.py"),
                "--run-id",
                args.run_id,
                "--port",
                str(args.port + 1),
                "--freeze-epochs",
                "2",
            ],
        )
        replay_server = subprocess.Popen(
            [
                sys.executable,
                str(ROOT / "scripts" / "reviewer_console_server.py"),
                "--run-id",
                args.run_id,
                "--port",
                str(args.port),
            ],
        )
        try:
            time.sleep(3.0)
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "capture_reviewer_screenshots.py"),
                    "--port",
                    str(args.port),
                    "--mid-port",
                    str(args.port + 1),
                ],
                check=True,
            )
        finally:
            replay_server.terminate()
            mid_server.terminate()
            replay_server.wait(timeout=5)
            mid_server.wait(timeout=5)

    zip_path = build_zip(OUT)

    docs_dir = ROOT / "docs" / "reviewer_submission"
    docs_dir.mkdir(parents=True, exist_ok=True)
    for name in ("SUPPLEMENTARY_MATERIAL.md", "COVER_LETTER.md", "README.md"):
        shutil.copy2(OUT / name, docs_dir / name)

    print(f"Built {OUT}")
    print(f"Zip: {zip_path}")


if __name__ == "__main__":
    main()
