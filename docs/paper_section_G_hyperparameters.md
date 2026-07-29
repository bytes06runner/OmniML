# How to add Section G (Hyperparameter Configuration and Search Space)

This file is a complete handoff note. Anyone can read it, understand Prof Tridib's request, and update the research paper correctly.

---

## 1. What Prof Tridib asked for

From the OmniML WhatsApp group, Prof said:

> add this section G. Hyperparameter Configuration and Search Space  
> there add this values

Meaning:

1. Add a new paper subsection titled **G. Hyperparameter Configuration and Search Space**.
2. Put the exact hyperparameter values the team already reported to him (sklearn Path B + optional PyTorch Path A).
3. Make it clear that the default tabular path uses scikit-learn, so epochs / learning rate / batch size do **not** apply there.

This is only one of several paper tasks he assigned. Other tasks (Visio figures, runtime-architecture style, LangGraph screenshot) are **not** covered by this file.

---

## 2. Which paper file to edit

Edit:

`C:\Users\mukhe\Downloads\revised_paper_figures_redrawn.docx`

Do **not** edit this Markdown file into the PDF directly. This Markdown is the instruction + paste source only.

After editing Word, export a fresh PDF for review/submission.

---

## 3. Exact placement in the paper

Target chapter: **V. EVALUATION PROTOCOL**

Current subsection order:

| Letter | Title |
| --- | --- |
| V-A | Evaluation Objectives |
| V-B | Dataset Selection and Cross-Modal Evaluation |
| V-C | Baseline Selection Strategy |
| V-D | Evaluation Metrics |
| V-E | Cross-Validation and Statistical Validation |
| V-F | Computational Environment and Reproducibility |
| V-G | Ablation Evaluation Strategy |

Required change:

1. Insert the new section **immediately after V-F**.
2. The new section becomes **V-G**.
3. Rename the old Ablation section from **V-G** to **V-H**.

Do **not** renumber III-G (Runtime Architecture) or IV-G (Comparative Benchmarking).

---

## 4. What to paste (recommended)

Use this version. It has no new numbered table, so you do **not** need to renumber Tables II and III.

### G. Hyperparameter Configuration and Search Space

The framework supports two execution paths with separate hyperparameter configurations. The default tabular path (Path B) uses scikit-learn estimators and performs an exhaustive seven-candidate grid search. Six candidates are generated for the Random Forest classifier from the Cartesian product of max_depth in {4, 8, unlimited} and n_estimators in {80, 150}. The seventh candidate is Logistic Regression with regularization parameter C = 1.0. Each candidate is evaluated on a stratified 80/20 training-validation split, and the configuration with the highest validation accuracy is selected for final fitting. All applicable data splitting, resampling, and Random Forest operations use a fixed random seed of 42.

For the reported tabular experiment, the selected configuration was Random Forest with max_depth = 8 and n_estimators = 150. Epoch count, learning rate, optimizer, and batch size are not applicable to Path B because the evaluated estimators are conventional scikit-learn models rather than gradient-trained neural networks.

The optional deep-learning path (Path A) compiles the human-approved architecture graph as a PyTorch model. Its four-candidate grid is formed from learning_rate in {0.001, 0.01} and batch_size in {32, 64}. The reported configuration uses five epochs, the Adam optimizer, and a fixed seed of 42. The two search spaces are reported separately to preserve the distinction between the default estimator-selection experiment and the optional neural execution path.

---

## 5. Step-by-step checklist

1. Open `revised_paper_figures_redrawn.docx`.
2. Find **V-F. Computational Environment and Reproducibility**.
3. Place the cursor after that subsection ends.
4. Paste the recommended Section G text from Section 4 above.
5. Rename existing **Ablation Evaluation Strategy** from **G** to **H**.
6. Format the new heading to match other V-* subsection headings.
7. Check that math/code symbols render cleanly in Word (max_depth, n_estimators, C, learning_rate, batch_size).
8. Save Word.
9. Export PDF.
10. Spot-check: Section V now contains A–H, and the hyperparameter values match the WhatsApp reply to Prof.

---

## 6. Values that must appear (verified)

### Path B — default tabular (scikit-learn)

| Item | Value |
| --- | --- |
| Estimators | Random Forest, Logistic Regression |
| RF `max_depth` | 4, 8, unlimited |
| RF `n_estimators` | 80, 150 |
| LogReg `C` | 1.0 |
| Search size | 7 candidates (6 RF + 1 LogReg) |
| Split | Stratified 80/20 |
| Seed | 42 |
| Selected config | RF `max_depth=8`, `n_estimators=150` |
| Not applicable | epochs, learning rate, optimizer, batch size |

### Path A — optional PyTorch

| Item | Value |
| --- | --- |
| Learning rate | 0.001, 0.01 |
| Batch size | 32, 64 |
| Epochs | 5 |
| Optimizer | Adam |
| Seed | 42 |
| Search size | 4 candidates |

Code sources:

- `anomallm/hpo.py`
- `anomallm/engineer.py`
- `anomallm/pytorch_engineer.py`
- `experiments/output/reviewer_share/evaluation_report.md`

---

## 7. Alternative: table version (only if requested)

Use this only if Prof / editor specifically wants a numbered table. It forces Table renumbering.

### G. Hyperparameter Configuration and Search Space

The framework supports two execution paths with separate hyperparameter configurations. The default tabular path (Path B) evaluates seven scikit-learn candidates through exhaustive grid search, whereas the optional deep-learning path (Path A) evaluates four PyTorch configurations. Table II reports the complete search space. Candidate configurations are evaluated on a stratified 80/20 training-validation split, and the configuration with the highest validation accuracy is selected for final fitting. A fixed seed of 42 is used for reproducibility.

**TABLE II — HYPERPARAMETER CONFIGURATION AND SEARCH SPACE**

| Execution path | Component | Parameter | Candidate value(s) |
| --- | --- | --- | --- |
| Path B (default) | Random Forest | max_depth | 4, 8, unlimited |
| Path B (default) | Random Forest | n_estimators | 80, 150 |
| Path B (default) | Logistic Regression | C | 1.0 |
| Path B (default) | Evaluation | Split and selection | Stratified 80/20; highest validation accuracy |
| Path B (default) | Reproducibility | Random seed | 42 |
| Path A (optional) | PyTorch | Learning rate | 0.001, 0.01 |
| Path A (optional) | PyTorch | Batch size | 32, 64 |
| Path A (optional) | PyTorch | Epochs | 5 |
| Path A (optional) | PyTorch | Optimizer | Adam |
| Path A (optional) | Reproducibility | Random seed | 42 |

Selected Path B result: Random Forest with max_depth = 8 and n_estimators = 150. Epochs / learning rate / optimizer / batch size are not applicable to Path B.

### Extra edits required for the table version

1. Current Table II (performance) → Table III.
2. Current Table III (ablation) → Table IV.
3. Section VI-A: “Table II summarises…” → “Table III summarises…”
4. Section III-D: “(Table III)” → “(Table IV)”
5. Section IV-E: “(Table III)” → “(Table IV)”
6. Section VI-C: “(Table III)” → “(Table IV)”

Default recommendation: **do not use the table version** unless asked.

---

## 8. Done criteria

You are done when all of the following are true:

- [ ] Paper Word file contains **V-G. Hyperparameter Configuration and Search Space**
- [ ] Ablation section is now **V-H**
- [ ] Path B values match the table in Section 6
- [ ] Selected RF config is `max_depth=8`, `n_estimators=150`
- [ ] Path A optional values are present
- [ ] Text explicitly says epochs/lr/batch size do not apply to Path B
- [ ] Fresh PDF exported and checked
