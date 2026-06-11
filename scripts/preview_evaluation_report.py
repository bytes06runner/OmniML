import json
from pathlib import Path

from anomallm.evaluation_report import render_evaluation_sections, search_space_table_rows
from anomallm.hpo import tabular_search_space

run = Path("runs/breast_cancer_biopsy_c795d4ae")
eval_data = json.loads((run / "artifacts/evaluation.json").read_text(encoding="utf-8"))
manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
tc = manifest["training_artifacts"]["training_config"]
space = tabular_search_space("classification")
arch = (
    "Dense(128,relu) -> BN1d -> Dropout(0.3) -> Dense(64,relu) -> "
    "BN1d -> Dropout(0.3) -> Dense(32,relu) -> Output(1,sigmoid)"
)
report = render_evaluation_sections(
    metrics=eval_data["metrics"],
    architecture=arch,
    training_config=tc,
    hpt_search_space=space,
    epoch_metrics=[{"epoch": 5}],
    best_params={"max_depth": 4, "n_estimators": 80},
)
print(report)
print("\n--- TRAINING CONSOLE JSON PREVIEW ---")
acc = eval_data["metrics"]["accuracy"]
strategy = f"{space['strategy']} ({len(space['candidates'])} candidates)"
console = {
    "status": "complete",
    "architecture": arch,
    "optimizer": tc["optimizer"],
    "search_strategy": strategy,
    "evaluation_protocol": {
        "accuracy_pct": round(acc * 100, 1),
        "f1": round(eval_data["metrics"]["f1"], 3),
        "epochs_executed": 5,
        "epochs_configured": tc["epochs"],
        "optimizer": tc["optimizer"],
        "architecture": arch,
        "search_strategy": strategy,
    },
    "search_space_table": [
        {"parameter": p, "search_range": r} for p, r in search_space_table_rows(space, tc)
    ],
}
print(json.dumps(console, indent=2))
