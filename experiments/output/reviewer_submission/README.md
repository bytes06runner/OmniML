# Reviewer Submission Pack

Authentic artifacts for post-acceptance reviewer requests.

## Contents

- `SUPPLEMENTARY_MATERIAL.pdf` — send with email
- `COVER_LETTER.md` — email body template
- `cv_results/` — 5-fold CV metrics (20260608_211634)
- `run_artifacts/breast_cancer_biopsy_c795d4ae/` — Chainlit run audit trail
- `screenshots/` — Training Console replay from real training.log

## Reproduce CV

```bash
python -m experiments.run --dataset breast_cancer --folds 5 --seed 42
python -m experiments.compare --latest
```

## Reproduce console screenshots

```bash
python scripts/reviewer_console_server.py --run-id breast_cancer_biopsy_c795d4ae
python scripts/capture_reviewer_screenshots.py
```

> Supersedes the staged demo pack in `experiments/output/reviewer_share/`.
