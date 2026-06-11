# Superseded

This folder contained **staged demo screenshots** (mock 96.5% metrics) for early UI review.

**Use instead:** [`../reviewer_submission/`](../reviewer_submission/)

That pack is built from:
- Real 5-fold CV (`experiments/run.py`)
- Run `breast_cancer_biopsy_c795d4ae` artifacts
- Training Console replay from `training.log`

Rebuild with:

```bash
python scripts/build_reviewer_submission.py
```
