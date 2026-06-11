# OmniML Autonomous ML Run Report

## Architecture Selection

- User query: Diagnose breast cancer from biopsy records
- Architecture: Dense(128,relu) -> BN1d -> Dropout(0.3) -> Dense(64,relu) -> BN1d -> Dropout(0.3) -> Dense(32,relu) -> Output(1,sigmoid)
- Dataset: utkarshx27/breast-cancer-wisconsin-diagnostic-dataset

## Evaluation Protocol

| Field | Value |
| --- | --- |
| Accuracy | 96.5% |
| F1 | 0.964 |
| Epochs | 5 executed / 5 configured |
| Optimizer | Adam |
| Architecture | Dense(128,relu) -> BN1d -> Dropout(0.3) -> Dense(64,relu) -> BN1d -> Dropout(0.3) -> Dense(32,relu) -> Output(1,sigmoid) |
| Search Strategy | grid (7 candidates) |
| Selected Hyperparameters | `{'max_depth': 8, 'n_estimators': 150}` |

## Hyperparameter Search Space

| Parameter | Search Range |
| --- | --- |
| Model / Estimator | logreg, rf |
| Regularization (C) | 1.0 |
| Max Depth | 4, 8, unlimited |
| N Estimators | 80, 150 |

## Explainability

Top features and SHAP/LIME artifacts are attached in the run folder under `runs/<run_id>/plots/`.

## Verdict

Evidence-backed run summary generated from run-scoped artifacts.
