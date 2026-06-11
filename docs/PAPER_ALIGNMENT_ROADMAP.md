# OmniML ↔ Research Paper Alignment Roadmap

> **Audience:** Human maintainers and AI coding agents (Cursor, Codex, etc.)  
> **Purpose:** Single source of truth for aligning the **OmniML** codebase with the research manuscript.  
> **Paper file (local):** `main (5).pdf` at repo root (title: *A Multi-Agent Orchestration Framework for Explainable Human-in-the-Loop AutoML*).  
> **Last updated:** 2026-06-04 (Path A: PyTorch architect-graph training, dual-path)

---

## How AI agents should use this document

1. **Read this file first** before implementing paper-related features.
2. **Pick one phase** (e.g. `1.2`) and complete its acceptance criteria before starting dependent phases.
3. **Update the [Progress tracker](#progress-tracker)** when a phase is done (change `[ ]` → `[x]` and add `Completed: YYYY-MM-DD` on the same line or in Notes).
4. **Do not over-claim in user-facing text** until the linked phase is marked complete (especially SHAP/LIME, baselines, cross-modal).
5. **Minimize scope:** match existing patterns in `graph.py`, `anomallm/*`, Chainlit `app.py`; avoid unrelated refactors.
6. **No commits** unless the user explicitly asks.

### Agent quick links

| Topic | Primary files |
|-------|----------------|
| LangGraph pipeline | `graph.py`, `app.py` |
| Training codegen | `anomallm/engineer.py` |
| XAI (current) | `anomallm/nodes.py` (`xai_node`) |
| XAI (legacy / unrelated) | `anomallm/explainer.py` (Granger — **not** paper SHAP/LIME) |
| Compliance | `anomallm/compliance.py` |
| Benchmarking (literature) | `anomallm/benchmarking.py`, `arxiv_comparator` in `graph.py` |
| Schemas / manifest | `anomallm/schemas.py`, `anomallm/runtime.py`, `anomallm/persistence.py` |
| HITL UI | `app.py`, `public/react_flow_editor/` |
| Dependencies | `requirements.txt`, `setup.py` |

---

## Executive summary (current gap)

The repo is a **strong match** for the paper’s **orchestration + HITL + compliance** story (LangGraph, shared state, run evidence bundles). **Dual-path training** is implemented: Path B (sklearn, default) and Path A (PyTorch MLP from approved architect graph, opt-in). Tabular SHAP/LIME apply to Path B; Path A records honest `pytorch_limited` XAI. Remaining gaps: real IMDB/CNN paper numbers, multi-dataset Table II batch runner, and guaranteed AutoKeras/H2O in CI.

**Rule for agents:** Implementation in this repo is the **source of truth** for behavior; the progress tracker below reflects what is shipped.

---

## Paper ↔ product naming

| Name | Where |
|------|--------|
| **OmniML** | `README.md`, user-facing branding |
| **AnomaLLM v3** | `graph.py` header comment (legacy — normalize in Wave 0) |
| **Proposed framework** (paper) | Generic; maps to OmniML implementation instance |

---

## Formal framework mapping (Paper §III)

Paper: \(F = (A, G, H, X, C, S)\)

| Symbol | Paper meaning | Code implementation | Status |
|--------|---------------|---------------------|--------|
| **G** | DAG orchestration graph | `graph.build_graph()` — LangGraph `StateGraph` | ✅ Implemented |
| **S** | Shared orchestration state | `AgentState` in `graph.py` + `EvidenceBundle` / `runs/<id>/manifest.json` | ✅ Implemented |
| **A** | Specialized agents \(a_{arch}, a_{eda}, a_{imb}, a_{opt}, a_{xai}, a_{bench}, a_{comp}\) | LangGraph **nodes** (mix of LLM + procedural) | ✅ Implemented (Path B default; Path A opt-in) |
| **H** | HITL at architecture + optimization | Architecture + `hitl_hpt_pause` + dataset/drift pauses | ✅ Implemented |
| **X** | SHAP global + LIME local | `anomallm/xai.py` — tabular + text/image branches | ✅ Implemented (Path B) |
| **C** | Compliance narratives | `anomallm/compliance.py` — EU AI Act, FDA SaMD, SOC2 | ✅ Implemented |

### Paper agent roster → code nodes

| Paper agent | Role | Code node(s) | Status |
|-------------|------|--------------|--------|
| \(a_{arch}\) | Architecture synthesis | `architect`, `hitl_model_pause` | ✅ JSON graph via Groq; **Path A** compiles graph to PyTorch when `training_path=pytorch` |
| \(a_{eda}\) | EDA | `eda_analyzer`, `hitl_eda_pause` | ✅ Tabular-focused |
| \(a_{imb}\) | Imbalance handling | `imbalance`, `anomallm/imbalance.py` | ✅ SMOTE, ADASYN, class_weight, focal-inspired weights |
| \(a_{opt}\) | HPO | `hpt`, `hitl_hpt_pause`, `anomallm/hpo.py` | ✅ Grid + optimization HITL |
| \(a_{xai}\) | SHAP + LIME | `xai_node`, `anomallm/xai.py` | ✅ SHAP summary + LIME JSON |
| \(a_{bench}\) | vs baselines / literature | `arxiv_comparator` + `experiments/` | ⚠️ Offline AutoKeras/H2O optional; ArXiv in UI |
| \(a_{comp}\) | Compliance | `compliance_mapper`, `compliance_narrative`, `compliance_renderer` | ✅ |

### Extra code nodes (not central in paper)

`run_history`, `kaggle_sourcer`, `dataset_ranker`, `dataset_downloader`, `dataset_validation`, `drift_sentry`, `hitl_drift_approval`, `modality`, `execution_choice`, `engineer`, `groq_loopfixer`, `codegen_contract_validator`, `execution_sandbox`, `debugger`, `model_deployer`, `evaluator`, `fairness_auditor`, `evidence_collector`, `compare_runs`, `save_run_history`

---

## Runtime flow (paper Fig. 1–3 vs code)

### Paper (simplified)

```text
P → Task abstraction → EDA → Architecture → HITL → Imbalance → HPO → HITL → Train → XAI → Benchmark → Compliance → Report
```

### Code (actual `graph.py` edges)

```text
run_history → architect → hitl_model_pause → kaggle_sourcer → dataset_ranker → hitl_pause
→ dataset_downloader → dataset_validation → [drift_sentry → hitl_drift_approval?]
→ modality → eda_analyzer → imbalance → hitl_eda_pause → execution_choice
→ hpt → engineer → [groq_loopfixer / validators] → execution_sandbox
→ xai_node → arxiv_comparator → model_deployer → evaluator → fairness_auditor
→ evidence_collector → compliance_* → save_run_history → compare_runs → END
```

### HITL checkpoints

| # | Paper | Code | UI / mechanism |
|---|-------|------|----------------|
| 1 | Architecture | ✅ | `hitl_model_pause` + React Flow editor (`public/react_flow_editor/`) |
| 2 | Optimization (pre-train) | ✅ | `hitl_hpt_pause` approves HPO params before `engineer` |
| — | (not in paper) | ✅ | Dataset selection `hitl_pause` |
| — | (not in paper) | ✅ | Drift `hitl_drift_approval` |

---

## Critical misalignments (remaining)

| Issue | Status | Notes |
|-------|--------|-------|
| **Architect graph ≠ trained model** | Open (Path B) | ADR: React Flow is design; sklearn template trains — Path A deferred |
| **Cross-modal UI** | Done (Path B) | Builtin + uploads; `modality_prepare` featurizes to CSV for sklearn |
| **Proxy datasets** | Partial | 20newsgroups TF-IDF as IMDB; flattened-pixel CIFAR — label in Table II |
| **AutoKeras/H2O in CI** | Optional | `skipped_*` when deps missing |
| **Neural focal loss** | Partial | Focal-inspired `sample_weight` on sklearn only (Wave 2.5) |
| **Granger vs paper XAI** | Mitigated | Banner on `anomallm/explainer.py` (Wave 1.6) |

**Fixed (do not re-open):** real SHAP/LIME (`anomallm/xai.py`), `feature_importance.png` naming, imbalance agent, `experiments/run.py` 5-fold + Table II.

---

## Technology stack (implementation truth)

| Layer | Paper (typical) | OmniML (actual) |
|-------|-----------------|-----------------|
| Orchestration | LangGraph | `langgraph` ✅ |
| LLM | Generic high-capacity LLM | **Groq** `openai/gpt-oss-120b` (`GROQ_MODEL`) |
| UI | Unspecified | **Chainlit** + embedded dashboards |
| Data | UCI BC, IMDB, CIFAR-10 | **Kaggle** (+ HF tools in `tools.py`) |
| Training (default) | Neural + HPO | **sklearn** + Optuna in generated script |
| Training (optional) | — | PyTorch compilation helpers in `graph.py` (`_build_model_class`) |
| XAI libs | SHAP, LIME | `shap`, `lime` in `requirements.txt` |
| Compliance | EU AI Act, FDA SaMD | Templates in `compliance.py` ✅ |
| Run storage | Persistent state | `runs/<run_id>/`, SQLite `persistence.py` |

---

## Evaluation claims vs repo (Paper §V–VI)

### Datasets (paper)

| Dataset | Modality | In repo today | Target phase |
|---------|----------|---------------|--------------|
| UCI Breast Cancer Wisconsin | Tabular | Via Kaggle workflows; not necessarily UCI split | 4.1, 5.2 |
| IMDB Sentiment | Text | ⚠️ Proxy (20newsgroups TF-IDF) | 5.3 |
| CIFAR-10 | Image | ⚠️ Offline flattened-pixel proxy | 5.4 |

### Baselines (paper)

| Baseline | Paper Table II | In repo |
|----------|----------------|---------|
| AutoKeras | ✅ | ⚠️ Optional (`experiments/baselines/sklearn_baselines.py`) |
| H2O AutoML | ✅ | ⚠️ Optional (graceful skip) |

### Metrics (paper §V-D)

- Accuracy, Macro F1, AUC-ROC — ✅ in offline `experiments/output/*/metrics.json`.
- Report as **μ ± σ** with **95% CI** over **5 folds** — ✅ via `python -m experiments.run --folds 5` (Chainlit engineer path may still use a smaller internal CV).

### Ablation (paper Table III)

| Config | Paper | Repo flag (planned Wave 0.4) |
|--------|-------|------------------------------|
| Full framework | ✅ | `enable_hitl=1`, `enable_xai=1`, multi-agent |
| Without HITL | ✅ | `enable_hitl=0` |
| Without explainability | ✅ | `enable_xai=0` |
| Without multi-agent orchestration | ✅ | `monolithic_mode=1` |

---

## Waves and phases (implementation backlog)

### Dependency graph

```text
Wave 0 (foundations)
  ├─→ Wave 1 (XAI)
  ├─→ Wave 2 (imbalance/HPO/HITL)
  └─→ Wave 3 (architecture fidelity)
        └─→ Wave 4 (experiments) ← also needs Wave 1–2 for honest tables
              ├─→ Wave 5 (cross-modal)
              └─→ Wave 6 (paper/manuscript sync)
```

---

## Wave 0 — Foundations & honest labeling

**Goal:** Safe baseline for agents; no false SHAP/baseline claims.

| ID | Task | Files to touch | Acceptance criteria |
|----|------|----------------|---------------------|
| 0.1 | Keep this roadmap current | `docs/PAPER_ALIGNMENT_ROADMAP.md` | Progress tracker updated per PR |
| 0.2 | Rename misleading plot keys until real SHAP | `anomallm/engineer.py`, `graph.py`, manifests, PDF report code | Use `feature_importance.png`; compliance/PDF text says “feature importance” unless Wave 1 done |
| 0.3 | Branding cleanup | `graph.py` header, `chainlit.md`, comments | Prefer **OmniML** in user-facing strings |
| 0.4 | Experiment / ablation flags | New `anomallm/config.py` or env vars; `graph.py` conditional edges | `OMNIML_ENABLE_HITL`, `OMNIML_ENABLE_XAI`, `OMNIML_MONOLITHIC` documented in `.env.example` |
| 0.5 | Smoke tests | `tests/test_graph_smoke.py`, `tests/test_compliance.py` | `pytest` passes: graph compiles; compliance builds for mock bundle |

---

## Wave 1 — Explainability (SHAP + LIME)

**Goal:** Paper §III-E, §IV-F — \(X(M,D) = \{X_g, X_l\}\).

| ID | Task | Files to touch | Acceptance criteria |
|----|------|----------------|---------------------|
| 1.1 | Dependencies | `requirements.txt`, `setup.py` | `shap`, `lime` (or `lime-tabular`) pinned |
| 1.2 | `anomallm/xai.py` module | **New** `anomallm/xai.py` | Functions: `compute_global_shap`, `compute_local_lime`; handle tabular sklearn models |
| 1.3 | Wire into pipeline | `anomallm/nodes.py`, `anomallm/engineer.py` OR post-train only in `xai_node` | Artifacts: `plots/shap_summary.png`, `artifacts/local_lime.json` |
| 1.4 | Schema + manifest | `anomallm/schemas.py` (`XAIArtifacts`) | Fields: `global_shap`, `local_lime`, `explanation_method` |
| 1.5 | Compliance + PDF | `anomallm/compliance.py`, `graph.py` evaluator/PDF | Narratives reference real artifacts; validation checks paths exist |
| 1.6 | Isolate Granger explainer | `anomallm/explainer.py` → `anomallm/anomaly/granger.py` (or docstring banner) | Agents don't confuse with XAI agent |

---

## Wave 2 — Imbalance & optimization

**Goal:** Paper §IV-D — ratio \(r\), SMOTE/ADASYN/focal/class weights; optimization HITL.

| ID | Task | Files to touch | Acceptance criteria |
|----|------|----------------|---------------------|
| 2.1 | Imbalance analysis | `anomallm/nodes.py`, optional `anomallm/imbalance.py` | State: `imbalance: {ratio, n_major, n_minor, recommended_strategy}` |
| 2.2 | Apply in training | `anomallm/engineer.py` template | `imbalanced-learn` when \(r <\) threshold; record `applied_strategy` in `evaluation.json` |
| 2.3 | HPO transparency | `graph.py` `hpt_node`, manifest | `theta_star` / `hpt_best_params` on manifest; search space documented |
| 2.4 | Optimization HITL | `graph.py`, `app.py`, new UI or extend training config | New interrupt **after** `hpt`, **before** `engineer`; user approves params |
| 2.5 | ADASYN + focal (Path B) | `anomallm/imbalance.py`, `anomallm/engineer.py` | `recommended_strategy` includes `adasyn` / `focal`; `evaluation.json` records `applied_strategy` |

---

## Wave 3 — Architecture fidelity

**Goal:** Paper §IV-B — \(G_a = \psi(T)\) executed or explicitly dual-path.

| ID | Task | Files to touch | Acceptance criteria |
|----|------|----------------|---------------------|
| 3.0 | **Decision** Path A vs B | This doc + `README.md` | Document chosen path in [Architecture decision](#architecture-decision-record) |
| 3.1 | Task abstraction | New node `task_abstraction` or extend `architect` | `task_representation: {modality, objective, constraints}` in state/manifest |
| 3.2a | **Path A** — Train compiled graph | `graph.py` `_build_model_class`, `engineer` PyTorch path | Approved `graph_architecture_json` drives `nn.Module` training |
| 3.2b | **Path B** — Tabular fast path | `README.md`, paper matrix | Paper/README state architect is **design**; sklearn path is **execution** for tabular |

---

## Wave 4 — Evaluation harness (Tables II & III)

**Goal:** Reproducible paper numbers from repo commands.

| ID | Task | Files to touch | Acceptance criteria |
|----|------|----------------|---------------------|
| 4.1 | Package layout | **New** `experiments/__init__.py`, `run.py`, `metrics.py`, `config.yaml` | CLI: `python -m experiments.run --dataset breast_cancer --folds 5` |
| 4.2 | Metrics | `experiments/metrics.py` | accuracy, macro_f1, auc_roc; output μ, σ, 95% CI |
| 4.3 | Baselines | `experiments/baselines/autokeras.py`, `h2o.py` | Optional extras; same seeds/splits |
| 4.4 | Table generator | `experiments/compare.py` | Writes `experiments/output/table_ii.md` (and CSV) |
| 4.5 | Ablation | Uses Wave 0.4 flags | Rows match Table III configs |
| 4.6 | README section | `README.md` | “Reproducing paper results” with exact commands |

---

## Wave 5 — Cross-modal

**Goal:** Paper §V-B — tabular + text + image.

| ID | Task | Files to touch | Acceptance criteria |
|----|------|----------------|---------------------|
| 5.1 | Modality routing | `graph.py` conditional from `modality_node` | Sub-flows or swappable engineer templates |
| 5.2 | UCI Breast Cancer loader | `experiments/datasets/uci_breast_cancer.py` | `--dataset uci_breast_cancer` works |
| 5.3 | IMDB | `experiments/datasets/imdb.py`, text engineer template | End-to-end run + experiment metrics |
| 5.4 | CIFAR-10 | `experiments/datasets/cifar10.py`, image template | Same |
| 5.5 | Modality XAI | `anomallm/xai.py` | Branches for text/image explainability |
| 5.6 | Chainlit cross-modal UI | `anomallm/featurize.py`, `modality_prepare`, `app.py` | Builtin + uploads; E2E sklearn on featurized CSV |

---

## Wave 6 — Manuscript & evidence sync

| ID | Task | Acceptance criteria |
|----|------|---------------------|
| 6.1 | Update paper text to match implementation | HITL count, stack, datasets actually run |
| 6.2 | Auto appendix from manifest | Script or node exports evidence index per experiment run |
| 6.3 | Limitations synced | ONNX placeholder, Groq dependency, etc. |

---

## Architecture decision record

| Decision | Options | Status | Chosen |
|----------|---------|--------|--------|
| **Training path** | **A:** Compile React Flow → PyTorch and train **B:** sklearn fast path on featurized CSV | ✅ Dual-path | **B default**, **A opt-in** via `OMNIML_TRAINING_PATH` or `training_config.training_path` (2026-06-04) |
| **Default data for paper repro** | UCI vs Kaggle breast cancer | ✅ Chosen | `breast_cancer` / `uci_breast_cancer` via sklearn (experiments) |
| **Baseline execution** | Local install vs Docker for H2O/AutoKeras | ✅ Chosen | Optional pip install; graceful skip in CI |

_Agents: Path B remains default; enable Path A only when the user or env requests `pytorch`._

---

## Progress tracker

Update checkboxes when acceptance criteria are met.

### Wave 0

- [x] **0.1** Roadmap maintained
- [x] **0.2** Honest plot naming
- [x] **0.3** Branding cleanup
- [x] **0.4** Ablation flags
- [x] **0.5** Smoke tests

### Wave 1

- [x] **1.1** SHAP/LIME deps
- [x] **1.2** `anomallm/xai.py`
- [x] **1.3** Pipeline wiring
- [x] **1.4** Schema updates
- [x] **1.5** Compliance/PDF
- [x] **1.6** Granger isolation (docstring banner; no file move)

### Wave 2

- [x] **2.1** Imbalance analysis
- [x] **2.2** Training strategies
- [x] **2.3** HPO transparency
- [x] **2.4** Optimization HITL
- [x] **2.5** ADASYN + focal-inspired weights (Path B)

### Wave 3

- [x] **3.0** Path A/B decision recorded (Path B)
- [x] **3.1** Task abstraction
- [x] **3.2b** Architecture execution (Path B sklearn doc)
- [x] **3.2a** Path A — train compiled PyTorch graph (`anomallm/graph_compile.py`, `anomallm/pytorch_engineer.py`)

### Wave 4

- [x] **4.1** Experiments CLI
- [x] **4.2** Metrics + CI
- [x] **4.3** Baselines
- [x] **4.4** Table II generator
- [x] **4.5** Ablation runs
- [x] **4.6** README repro section

### Wave 5

- [x] **5.1** Modality routing
- [x] **5.2** UCI loader
- [x] **5.3** IMDB (experiments)
- [x] **5.4** CIFAR-10 (Wave 5b)
- [x] **5.5** Modality XAI (text/image branches)
- [x] **5.6** Chainlit cross-modal UI (builtin + uploads, featurization path)

### Wave 6

- [x] **6.1** Claim-to-artifact map (below)
- [x] **6.2** Evidence appendix script
- [x] **6.3** Limitations sync (README)

---

## Suggested agent sprint order (post kick-off)

**Sprint 1 (done):** 2.3, 4.3–4.6, 3.2b  
**Sprint 2 (done):** 2.4, 3.1  
**Sprint 3 (done):** 5.1–5.3, 6.1–6.3  
**Next:** optional multi-dataset Table II batch runner; real HF IMDB / CNN baselines; DeepExplainer for Path A XAI (stretch)

---

## Claim-to-artifact map (§V–VI)

| Paper claim | Command / artifact |
|-------------|-------------------|
| 5-fold CV breast cancer | `python -m experiments.run --dataset breast_cancer --folds 5` → `experiments/output/*/metrics.json` |
| Table II comparison | `python -m experiments.compare --latest` → `experiments/output/table_ii.md` |
| SHAP + LIME | Chainlit run → `runs/<id>/plots/shap_summary.png`, `artifacts/local_lime.json` |
| Imbalance ratio + strategy | `runs/<id>/artifacts/evaluation.json` → `imbalance` (`smote`, `adasyn`, `class_weight`, `focal`) |
| HPO / θ\* | `runs/<id>/artifacts/hpt_summary.json`, manifest `metadata.theta_star` |
| Optimization HITL | UI pause `hitl_hpt_pause` + `is_hpt_approved` in state |
| Task abstraction φ(P) | `task_abstraction` node → `manifest.metadata.task_representation` |
| Compliance reports | `runs/<id>/reports/{eu_ai_act,fda_samd,soc2}.{md,html,pdf}` |
| Evidence appendix | `python scripts/export_evidence_appendix.py runs/<id>/manifest.json` |
| CIFAR-10 (flattened-pixel proxy) | `python -m experiments.run --dataset cifar10 --folds 5` → `experiments/output/*/metrics.json` |
| Text/image modality XAI | Chainlit run with `modality` text/image → `artifacts/xai_summary.json` (`explanation_method` contains `text_tfidf` or `image_flattened`) |
| Chainlit text/image E2E | Builtin `omniml/imdb-text-proxy` or `omniml/cifar10-image-proxy`, or upload → `modality_prepare` → `artifacts/features.csv` |
| Path A PyTorch (architect graph) | `training_config.training_path=pytorch` or `OMNIML_TRAINING_PATH=pytorch` → `runs/<id>/exports/model.pt` (state_dict), real `epoch_metric` stream |

---

## Coding conventions (for agents)

- **Python 3.12+**; match existing type hints and Pydantic models in `anomallm/schemas.py`.
- **Run artifacts** always under `runs/<run_id>/`; register via `register_artifact` in `anomallm/runtime.py`.
- **Do not** break Chainlit HITL: nodes in `UI_MANAGED_INTERRUPTS` must not call `interrupt()` internally.
- **LLM calls:** Groq via `langchain_groq` or `groq` SDK as in `architect_node`.
- **Tests:** prefer small fixtures under `tests/fixtures/`; no live Kaggle in unit tests.
- **Paper claims:** if a feature isn't done, UI/report copy must not imply it (see Wave 0.2).

---

## Key schema extensions (implemented)

`XAIArtifacts` in `anomallm/schemas.py` includes `global_shap`, `local_lime`, `plot_paths`, and `explanation_method` (Wave 1).

Orchestration state / manifest includes `imbalance: Dict[str, Any]` with `recommended_strategy`, `applied_strategy`, and warnings (Waves 2.1–2.5).

---

## References inside repo

| Resource | Path |
|----------|------|
| User README | `README.md` |
| Plugin SDK | `docs/plugin_sdk.md` |
| Paper PDF | `main (5).pdf` |
| Example run manifest | `runs/breast_cancer_biopsy_*/manifest.json` |
| Env template | `.env.example` |

---

## Notes / changelog

| Date | Author | Note |
|------|--------|------|
| 2026-06-04 | Initial roadmap | Created from paper vs codebase audit |
| 2026-06-04 | Kick-off sprint | Waves 0–1, 2.1–2.2, 4.1–4.2; Path B ADR; see `anomallm/config.py`, `anomallm/xai.py`, `experiments/run.py` |
| 2026-06-04 | Sprints 1–3 | 2.3–2.4, 3.1, 4.3–4.6, 5.1–5.3, 6.1–6.3; `anomallm/hpo.py`, `experiments/compare.py`, `hitl_hpt_pause` |
| 2026-06-04 | Wave 5b | 5.4 `experiments/datasets/cifar10.py`, multiclass proba fix; 5.5 `run_xai_for_modality` in `anomallm/xai.py` |
| 2026-06-04 | Sprint B | 2.5 ADASYN + focal-inspired weights; roadmap audit sync with tracker |
| 2026-06-04 | Cross-modal UI | 5.6 `anomallm/featurize.py`, `modality_prepare`, builtin/local downloads, Chainlit uploads |

_Add rows when phases complete or decisions are made._
