from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DATASET_LABELS: Dict[str, str] = {
    "breast_cancer": "UCI Breast Cancer",
    "uci_breast_cancer": "UCI Breast Cancer",
    "imdb": "IMDB (20newsgroups proxy)",
    "cifar10": "CIFAR-10 (flattened-pixel proxy)",
    "cifar_10": "CIFAR-10 (flattened-pixel proxy)",
}


def _dataset_label(dataset_id: str) -> str:
    return DATASET_LABELS.get(dataset_id, dataset_id)


def _latest_metrics_file(output_root: Path) -> Optional[Path]:
    candidates = sorted(output_root.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _row_from_summary(dataset: str, framework: str, summary: Dict[str, Any], status: str) -> Dict[str, str]:
    acc = summary.get("accuracy", {})
    f1 = summary.get("macro_f1", {})
    auc = summary.get("auc_roc", {})
    if status != "ok" or acc.get("mean") is None:
        return {
            "dataset": dataset,
            "framework": framework,
            "accuracy": "N/A",
            "macro_f1": "N/A",
            "auc_roc": "N/A",
            "status": status,
        }
    return {
        "dataset": dataset,
        "framework": framework,
        "accuracy": f"{acc['mean']*100:.1f} ± {acc['std']*100:.1f}",
        "macro_f1": f"{f1['mean']:.3f} ± {f1['std']:.3f}",
        "auc_roc": f"{auc['mean']:.3f} ± {auc['std']:.3f}",
        "status": status,
    }


def build_table_ii(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    dataset = _dataset_label(payload.get("dataset", "unknown"))
    rows = []
    for framework, block in payload.get("frameworks", {}).items():
        rows.append(_row_from_summary(dataset, framework, block.get("summary") or {}, block.get("status", "ok")))
    return rows


def build_table_iii(ablation_runs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    rows = []
    for run in ablation_runs:
        omniml = (run.get("frameworks") or {}).get("omniml", {})
        summary = omniml.get("summary") or {}
        acc = summary.get("accuracy", {})
        f1 = summary.get("macro_f1", {})
        rows.append(
            {
                "configuration": run.get("ablation", "full"),
                "accuracy": f"{acc.get('mean', 0)*100:.1f}" if acc.get("mean") is not None else "N/A",
                "macro_f1": f"{f1.get('mean', 0):.3f}" if f1.get("mean") is not None else "N/A",
                "status": omniml.get("status", "unknown"),
            }
        )
    return rows


def write_markdown_table(path: Path, title: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        path.write_text(f"# {title}\n\nNo rows.\n", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper comparison tables from experiment output")
    parser.add_argument("--latest", action="store_true", help="Use latest experiments/output/*/metrics.json")
    parser.add_argument("--metrics", default="", help="Explicit metrics.json path")
    args = parser.parse_args()

    output_root = Path(__file__).parent / "output"
    metrics_path = Path(args.metrics) if args.metrics else _latest_metrics_file(output_root)
    if not metrics_path or not metrics_path.exists():
        raise SystemExit("No metrics.json found. Run python -m experiments.run first.")

    with open(metrics_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    table_ii = build_table_ii(payload)
    table_iii = build_table_iii([payload])

    out_dir = output_root
    write_markdown_table(out_dir / "table_ii.md", "Table II — Comparative Performance", table_ii)
    with open(out_dir / "table_ii.csv", "w", encoding="utf-8", newline="") as handle:
        if table_ii:
            writer = csv.DictWriter(handle, fieldnames=list(table_ii[0].keys()))
            writer.writeheader()
            writer.writerows(table_ii)
    write_markdown_table(out_dir / "table_iii.md", "Table III — Ablation (single run)", table_iii)
    print(f"Wrote {out_dir / 'table_ii.md'}")


if __name__ == "__main__":
    main()
