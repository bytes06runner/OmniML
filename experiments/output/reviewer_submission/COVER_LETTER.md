# Cover Letter (email template)

Dear Editor and Reviewers,

Thank you for accepting our paper. As requested, we attach supplementary material documenting the **evaluation protocol** and **hyperparameter search space** used in our experiments.

**Primary reproducible results (5-fold CV, UCI breast cancer):**

- **accuracy**: 0.9631 ± 0.0073 (95% CI 0.9568 – 0.9695)
- **macro_f1**: 0.9605 ± 0.0076 (95% CI 0.9538 – 0.9672)
- **auc_roc**: 0.9913 ± 0.0074 (95% CI 0.9848 – 0.9977)
- **accuracy**: 0.9631 ± 0.0073 (95% CI 0.9568 – 0.9695)
- **macro_f1**: 0.9605 ± 0.0076 (95% CI 0.9538 – 0.9672)
- **auc_roc**: 0.9913 ± 0.0074 (95% CI 0.9848 – 0.9977)
- **accuracy**: 0.9508 ± 0.0252 (95% CI 0.9287 – 0.9730)
- **macro_f1**: 0.9475 ± 0.0268 (95% CI 0.9240 – 0.9710)
- **auc_roc**: 0.9923 ± 0.0050 (95% CI 0.9879 – 0.9967)


**Hyperparameter search:** 7-point grid over RandomForest (`max_depth` ∈ {4, 8, unlimited}, `n_estimators` ∈ {80, 150}) and LogisticRegression (`C` = 1.0), implemented in `anomallm/hpo.py`.

**Attached files:**
1. `SUPPLEMENTARY_MATERIAL.pdf` — full protocol, search space table, reproducibility commands
2. `reviewer_submission.zip` — CV metrics, run artifacts for `breast_cancer_biopsy_c795d4ae`, and Training Console screenshots replayed from real `training.log` JSON telemetry

**Important distinction:** Paper Table II numbers come from offline CV (`experiments/run.py`). Console screenshots document the HITL transparency UI (epochs, optimizer config, architecture, search space) from run `breast_cancer_biopsy_c795d4ae` on a single holdout split.

Best regards,
[Authors]
