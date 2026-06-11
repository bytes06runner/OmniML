# Experiment baselines

| Framework | Status | Notes |
|-----------|--------|-------|
| OmniML (sklearn grid) | Always available | `experiments/run.py` |
| sklearn_rf / sklearn_logreg | Always available | Same runner |
| AutoKeras | Optional | `pip install autokeras`; skipped in CI if missing |
| H2O AutoML | Optional | Requires Java + `h2o` package; skipped in CI if missing |

Install optional baselines locally when reproducing paper Table II:

```bash
pip install autokeras h2o
```
