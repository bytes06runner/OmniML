# OmniML Autonomous ML Run Report

## Architecture Selection

- User query: "Diagnose breast cancer from biopsy records"
- Architecture: Dense(128,relu) → BN1d → Dropout(0.3) → Dense(64,relu) → Dropout(0.3) → Dense(32,relu) → Output(1,sigmoid)
- Dataset: utkarshx27/breast-cancer-wisconsin-diagnostic-dataset

## Training Summary

- Metrics: No metrics recorded.
- Best params: No HPT params recorded.

## Explainability

The model explanation is evidence-backed and should be interpreted as directional guidance rather than causal proof. Top features reflect the most influential available signals from the current training artifacts.

## Benchmarking

## Benchmark Gap Analysis

- Task label: `medical_diagnosis_classification`
- Source status: `unavailable`
- Cache hit: `False`
- Retrieved at: `2026-04-09T07:10:46.307716+00:00`
- Comparability score: `0.2`
- Directly comparable: `False`

### Sources
- No leaderboard entries were available; related literature context only.

### Retrieval Failures
- Papers With Code retrieval failed: Expecting value: line 1 column 1 (char 0)

### Recommendations
- [heuristic] Match the benchmark dataset split and evaluation metric exactly before treating a benchmark gap as actionably real.

## Verdict

Evidence-backed run summary generated from run-scoped artifacts. Review compliance attachments for regulated deployment contexts.