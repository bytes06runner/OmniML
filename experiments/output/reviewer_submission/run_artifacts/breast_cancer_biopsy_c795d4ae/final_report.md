# OmniML Autonomous ML Run Report

## Architecture Selection

- User query: Diagnose breast cancer from biopsy records
- Architecture (HITL-approved graph): Dense(128,relu) -> BN1d -> Dropout(0.3) -> Dense(64,relu) -> BN1d -> Dropout(0.3) -> Dense(32,relu) -> Output(1,sigmoid)
- Actual estimator (Path B): RandomForest selected via grid search
- Dataset: utkarshx27/breast-cancer-wisconsin-diagnostic-dataset

## Evaluation Protocol

| Field | Value |
| --- | --- |
| Accuracy | 100.0% |
| F1 | 1.000 |
| Epochs | 5 executed / 5 configured |
| Optimizer | Adam |
| Architecture | Dense(128,relu) -> BN1d -> Dropout(0.3) -> Dense(64,relu) -> BN1d -> Dropout(0.3) -> Dense(32,relu) -> Output(1,sigmoid) |
| Search Strategy | grid (7 candidates) |
| Selected Hyperparameters | `{'max_depth': 4, 'n_estimators': 80}` |

## Hyperparameter Search Space

| Parameter | Search Range |
| --- | --- |
| Model / Estimator | logreg, rf |
| Regularization (C) | 1.0 |
| Max Depth | 4, 8, unlimited |
| N Estimators | 80, 150 |

## Explainability

The model explanation is evidence-backed and should be interpreted as directional guidance rather than causal proof. Top features reflect the most influential available signals from the current training artifacts.

## Verdict

Evidence-backed run summary generated from run-scoped artifacts. Paper Table II metrics use separate 5-fold CV via experiments/run.py.