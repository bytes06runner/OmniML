<p align="center">
  <img src="public/logo.png" alt="OmniML Logo" width="120" />
</p>

<h1 align="center">OmniML - Autonomous HITL Auto-ML Pipeline</h1>

<p align="center">
  <strong>Describe a problem. Review the architecture. Pick the dataset. Launch training. Download the reports.</strong>
</p>

<p align="center">
  <a href="#key-features">Features</a> •
  <a href="#current-flow">Current Flow</a> •
  <a href="#installation">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#artifacts-and-reports">Artifacts</a> •
  <a href="#project-structure">Structure</a>
</p>

---

**OmniML** is a LangGraph-orchestrated machine learning system with explicit human-in-the-loop checkpoints. It uses **Groq** (`openai/gpt-oss-120b`) for reasoning, **Chainlit** for the chat UI, and real dataset sourcing via **Kaggle** for the current tabular-first workflow.

The codebase now reflects a stabilized end-to-end flow:
- inline embedded HITL views owned by Chainlit
- structured dataset acquisition and validation state
- explicit execution-mode selection before sandbox execution
- run-scoped evidence bundles and downloadable compliance artifacts

## Key Features

### Multi-agent pipeline

OmniML currently runs a staged workflow across these major responsibilities:

| Stage | Role |
|---|---|
| Architect | Generates a baseline model graph from the user problem |
| Visual HITL Editor | Lets the user refine architecture in the embedded graph canvas |
| Dataset Sourcing | Searches, ranks, downloads, and validates real tabular datasets |
| EDA | Profiles the selected CSV and renders an embedded review dashboard |
| Training Config | Collects hyperparameters, compliance modes, fairness config, and benchmark settings |
| Compute Strategy | Forces an explicit Local vs Cloud execution choice before training resumes |
| Engineer / Loopfixer / Debugger | Generates deterministic training code and validates it before execution |
| Execution Sandbox | Runs the selected script and captures logs and metrics |
| Trust Pipeline | Produces benchmark, fairness, compliance, and deployment artifacts |

### Human-in-the-loop checkpoints

The current UI-managed pause barriers are:
1. architecture review
2. dataset selection
3. EDA review
4. execution-mode selection
5. drift approval when drift checking requires it

### Run-scoped evidence and downloads

Each run persists to `runs/<run_id>/manifest.json` as a full evidence bundle. The app normalizes that bundle for:
- run artifact browsing
- compliance report status
- benchmark status
- report downloads via `/dl-run-artifact/<run_id>/<kind>/<filename>`

Compliance outputs currently include:
- markdown report
- html report
- pdf report

### Deployment bundle

Successful runs can generate:
- `model.pt`
- `model_scripted.pt`
- `model.onnx`
- `model_meta.json`
- `serve_api.py`
- `Dockerfile`
- `requirements.txt`

These appear in the embedded deployment dashboard and are downloadable from the app.

## Current Flow

```text
User Query
  -> Architecture generation
  -> HITL architecture editor
  -> Dataset sourcing and ranking
  -> HITL dataset selection
  -> Dataset download + validation
  -> Drift sentry
  -> EDA profiling
  -> HITL training configuration
  -> HITL compute strategy
  -> HPT + deterministic code generation
  -> Code validation
  -> Execution sandbox
  -> XAI / benchmark / fairness / compliance
  -> Deployment dashboard + downloadable reports
```

## Path A vs Path B (training execution)

| Path | Default | How to enable | What runs |
|------|---------|---------------|-----------|
| **B (sklearn)** | Yes | `training_path: sklearn` in Training Config, or `OMNIML_TRAINING_PATH=sklearn` | Fast tabular/text/image training on featurized CSV via sklearn grid search |
| **A (PyTorch)** | No | `training_path: pytorch` in Training Config, or `OMNIML_TRAINING_PATH=pytorch` | Compiles the approved React Flow graph to `OmniMLNet` and trains with real epoch metrics |

Path A uses the same featurized CSV pipeline as cross-modal Path B (`features.csv` for text/image). UI SHAP/LIME remain sklearn-oriented; PyTorch runs record `explanation_method: pytorch_limited` in XAI artifacts.

Important current behavior:
- training does not begin until `execution_choice` is selected
- report download links resolve through run-scoped artifact refs, not hardcoded filesystem paths
- terminal codegen failure stops the run instead of continuing into execution
- compliance reports are persisted both as files on disk and as bundle metadata

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/bytes06runner/OmniML.git
cd OmniML
```

### 2. Create the environment

OmniML requires **Python 3.12+**.

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure `.env`

```bash
copy .env.example .env
```

Required variables:

```env
GROQ_API_KEY=your_groq_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
CHAINLIT_AUTH_SECRET=your_random_secret
```

Optional runtime note:
- `OMNIML_SKIP_KAGGLE_AUTH=1` skips the startup credential write path. This is mainly useful for tests or restricted environments.

### 4. Rebuild frontend assets only if needed

The React Flow editor bundle is already checked in. Rebuild only if you change the editor source:

```bash
cd public\react_flow_editor
npm install
npm run build
cd ..\..
```

## Usage

### Launch locally

```bash
python start.py
```

Then open [http://localhost:8001](http://localhost:8001).

### Example prompts

> "Diagnose breast cancer from biopsy records"

> "Build a model to predict housing prices from demographic data"

> "Detect credit card fraud from transaction telemetry"

### Runtime diagnostics

The app also exposes a lightweight diagnostics endpoint:

- [http://localhost:8001/runtime-diagnostics](http://localhost:8001/runtime-diagnostics)

It is useful for checking:
- Groq auth health
- Kaggle CLI resolution
- credential availability
- recent dataset download status

## Artifacts and Reports

### Run outputs

Run artifacts are stored under:

```text
runs/<run_id>/
  artifacts/
  plots/
  reports/
  exports/
  logs/
  manifest.json
```

### Download behavior

The app serves run-scoped artifacts from the manifest-backed route:

```text
/dl-run-artifact/<run_id>/<kind>/<filename>
```

Current report kinds:
- `report_markdown`
- `report_html`
- `report_pdf`

### Embedded dashboards

The current embedded views include:
- architecture editor
- pipeline status
- EDA dashboard
- training config
- training console
- HPT console
- deployment dashboard

## Paper alignment roadmap

For AI agents and maintainers aligning the codebase with the research paper (`main (5).pdf`), see **[docs/PAPER_ALIGNMENT_ROADMAP.md](docs/PAPER_ALIGNMENT_ROADMAP.md)** — waves, phases, claim matrix, file map, and progress tracker.

### Architecture execution (Path B)

The React Flow canvas is a **human-reviewed design artifact**. Tabular training executes via the deterministic **sklearn** template in `anomallm/engineer.py`. Neural graph compilation remains available for future Path A work.

### Reproducing offline experiment metrics

```bash
pip install -e .
python -m experiments.run --dataset breast_cancer --folds 5
python -m experiments.run --dataset uci_breast_cancer --folds 5
python -m experiments.run --dataset imdb --folds 5 --frameworks omniml,sklearn_rf
python -m experiments.run --dataset cifar10 --folds 5 --frameworks omniml,sklearn_rf
python -m experiments.compare --latest
```

CIFAR-10 uses a **flattened-pixel sklearn proxy** (not a CNN in the UI). Requires `torchvision` (`pip install torchvision`).

### Cross-modal Chainlit (text / image)

In the UI, describe a text or image task (e.g. "IMDB sentiment" or "CIFAR-10 classification"). Pick a **builtin** dataset:

- `omniml/imdb-text-proxy` — TF-IDF text features (20newsgroups proxy)
- `omniml/cifar10-image-proxy` — flattened CIFAR-10 pixels for sklearn

**Uploads** (attach a file to your message):

- Text: `.jsonl` with `text` + `label`, or `.txt` with `label<TAB>text` per line
- Image: `.zip` of images (class = subfolder name) or a single `.png`/`.jpg`

Training still uses Path B sklearn on featurized `features.csv`; XAI uses modality-specific branches.

Optional baselines (see `experiments/baselines/README.md`):

```bash
pip install autokeras h2o
python -m experiments.run --dataset breast_cancer --frameworks omniml,sklearn_rf,autokeras,h2o
```

Ablation flag:

```bash
python -m experiments.run --dataset breast_cancer --ablation monolithic
```

Outputs: `experiments/output/<timestamp>/metrics.json`, `experiments/output/table_ii.md`.

Evidence appendix from a Chainlit run:

```bash
python scripts/export_evidence_appendix.py runs/<run_id>/manifest.json
```

### Smoke tests

```bash
pytest tests/
```

### Known limitations (paper alignment)

- ONNX export is a placeholder bytes stub.
- AutoKeras/H2O baselines are optional and may report `skipped_*` in CI.
- Full Chainlit auto-bypass for `OMNIML_ENABLE_HITL=0` is not implemented; ablations use the offline runner.
- CIFAR-10 / IMDB in the UI use documented proxies (not full HF IMDB or in-app CNN).
- Text/image XAI explains TF-IDF or flattened feature CSVs; raw-token or CNN SHAP is not supported in-app.
- Imbalance handling (Path B): SMOTE, ADASYN, class weights, and focal-inspired sample weights for sklearn; not full neural focal loss.

## Plugin SDK

OmniML supports trusted in-process enterprise plugins. See [docs/plugin_sdk.md](docs/plugin_sdk.md) for the plugin manifest format, supported slots, and the example plugin under `plugins/example_evidence/`.

## Project Structure

```text
OmniML/
├── app.py
├── graph.py
├── tools.py
├── start.py
├── chainlit.md
├── README.md
├── public/
│   ├── custom.js
│   ├── react_flow_editor/
│   ├── eda_dashboard/
│   ├── training_config/
│   ├── training_console/
│   ├── hpt_console/
│   ├── pipeline_status/
│   └── deployment_dashboard/
├── anomallm/
│   ├── compliance.py
│   ├── runtime.py
│   ├── schemas.py
│   ├── persistence.py
│   ├── engineer.py
│   └── backends/
└── tests/
```

## Current Status

Implemented and working in the current codebase:
- structured dataset download and validation flow
- EDA handoff into training configuration
- explicit compute-strategy gate before execution
- deterministic code generation with bounded repair attempts
- run-scoped compliance report generation and downloads
- deployment dashboard downloads backed by normalized manifest reads

Still in cleanup / polish territory:
- broader UI text cleanup
- additional training-flow end-to-end coverage
- further reduction of startup side effects outside real app runs

---

<p align="center">
  <strong>OmniML</strong> - autonomous, evidence-backed, and HITL-controlled.
</p>
