# OmniML Supplementary Material

## 1. Evaluation Protocol

### 1.1 Paper metrics (5-fold cross-validation)

Primary Table II numbers are produced offline by `experiments/run.py` on the **sklearn UCI breast cancer** dataset (569 samples, 30 features, binary target).

# Experiment Summary

- Dataset: `breast_cancer`
- Folds: `5`
- Ablation: `full`

## omniml (ok)
- **accuracy**: 0.9631 ± 0.0073 (95% CI 0.9568 – 0.9695)
- **macro_f1**: 0.9605 ± 0.0076 (95% CI 0.9538 – 0.9672)
- **auc_roc**: 0.9913 ± 0.0074 (95% CI 0.9848 – 0.9977)

## sklearn_rf (ok)
- **accuracy**: 0.9631 ± 0.0073 (95% CI 0.9568 – 0.9695)
- **macro_f1**: 0.9605 ± 0.0076 (95% CI 0.9538 – 0.9672)
- **auc_roc**: 0.9913 ± 0.0074 (95% CI 0.9848 – 0.9977)

## sklearn_logreg (ok)
- **accuracy**: 0.9508 ± 0.0252 (95% CI 0.9287 – 0.9730)
- **macro_f1**: 0.9475 ± 0.0268 (95% CI 0.9240 – 0.9710)
- **auc_roc**: 0.9923 ± 0.0050 (95% CI 0.9879 – 0.9967)


**Reproduce:**

```bash
pip install -r requirements.txt
python -m experiments.run --dataset breast_cancer --folds 5 --seed 42 --frameworks omniml,sklearn_rf,sklearn_logreg
python -m experiments.compare --latest
```

### 1.2 Chainlit UI audit run (holdout evaluation)

Run ID: `breast_cancer_biopsy_c795d4ae`

| Field | Value |
| --- | --- |
| Protocol | Single 80/20 stratified holdout on Kaggle Wisconsin diagnostic CSV |
| Accuracy | 100.0% |
| F1 | 1.000 |
| Epochs | 5 configured (console epoch curves are telemetry for UI; Path B uses sklearn) |
| Optimizer (config) | Adam (UI config); actual search uses sklearn RF/LogReg grid |
| Selected estimator | RandomForest |
| Selected hyperparameters | `{'max_depth': 4, 'n_estimators': 80}` |

### 1.3 Architecture vs training path (transparency note)

- **HITL-approved graph:** Dense(128,relu) → BN1d → Dropout → Dense layers → Output (design artifact)
- **Default execution (Path B):** sklearn grid search on featurized CSV — this is what produced both CV benchmarks and the cited UI run estimator
- **Optional Path A:** Set `training_path=pytorch` to compile the approved graph to PyTorch
