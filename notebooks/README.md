# Reproducibility notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bytes06runner/OmniML/blob/main/notebooks/OmniML_Reproducibility.ipynb)

`OmniML_Reproducibility.ipynb` is the executable companion to *A Multi-Agent Orchestration
Framework for Explainable Human-in-the-Loop AutoML*. It implements the framework of paper
Sections III–IV and reproduces the experiments of Sections V–VI from a clean Colab runtime.

## Running it

1. Open the badge above, or upload the notebook to [colab.research.google.com](https://colab.research.google.com).
2. **Runtime → Change runtime type → T4 GPU.**
3. **Runtime → Run all.**

The first cell is the only one that needs editing:

| Setting | Default | Effect |
|---|---|---|
| `FULL_RUN` | `False` | `False` runs a ~5 minute verification pass. `True` runs the complete reproduction (~60–75 min on a T4): DistilBERT + LoRA on the real IMDB corpus, a CNN trained to convergence on full CIFAR-10, and the AutoKeras / H2O AutoML baselines. |
| `EXHAUSTIVE` | `False` | Raises the text and image experiments from 3 seed repeats to 5. |
| `RUN_BASELINES` | `True` | Installs and runs AutoKeras and H2O AutoML. |
| `GROQ_API_KEY` | `""` | Optional. With a key the reasoning agents run as real LLM calls; without one they fall back to deterministic implementations and every affected artifact is labelled accordingly. |
| `SEED` | `42` | Global seed. |

## What it produces

Everything is computed during execution — no results are transcribed from the paper. On
completion the notebook writes `omniml_results.zip` containing:

- `artifacts/table_ii_comparative.csv` — comparative performance across the three modalities
- `artifacts/table_iii_ablation.csv` — component ablation
- `artifacts/reproducibility_manifest.json` — configuration, package versions, hardware, seeds, timings, and all raw per-fold metrics
- `figures/` — orchestration DAG, comparative and ablation charts, per-run dispersion, SHAP global attribution, LIME local attribution, confusion matrix
- `reports/` — EU AI Act, FDA SaMD, and SOC 2 governance documents generated from the run's own artifacts

## Editing the notebook

The notebook is generated from `build_notebook.py`, which holds every cell as plain Python.
Notebook JSON is hard to review in a diff, so edits go through the generator:

```bash
python notebooks/build_notebook.py
```

Edit `build_notebook.py`, re-run it, and commit both files. Hand-editing the `.ipynb` directly
will be overwritten by the next regeneration.
