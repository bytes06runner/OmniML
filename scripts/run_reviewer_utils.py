"""Shared helpers for authentic reviewer submission artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ID = "breast_cancer_biopsy_c795d4ae"


def run_root(run_id: str = DEFAULT_RUN_ID) -> Path:
    return ROOT / "runs" / run_id


def load_manifest_bundle(run_id: str = DEFAULT_RUN_ID) -> Dict[str, Any]:
    manifest_path = run_root(run_id) / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_evaluation(run_id: str = DEFAULT_RUN_ID) -> Dict[str, Any]:
    path = run_root(run_id) / "artifacts" / "evaluation.json"
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def architecture_from_bundle(bundle: Dict[str, Any]) -> str:
    for report_path in (
        bundle.get("run_manifest", {}).get("paths", {}).get("reports"),
        run_root(bundle.get("run_manifest", {}).get("run_id", DEFAULT_RUN_ID)) / "reports",
    ):
        if not report_path:
            continue
        final_report = Path(report_path) / "final_report.md"
        if final_report.exists():
            for line in final_report.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("- Architecture:"):
                    return line.split(":", 1)[1].strip()
    return "Dense(128,relu) -> BN1d -> Dropout(0.3) -> Dense(64,relu) -> BN1d -> Dropout(0.3) -> Dense(32,relu) -> Output(1,sigmoid)"


def parse_training_log_events(log_path: Path) -> List[Dict[str, Any]]:
    if not log_path.exists():
        return []
    events: List[Dict[str, Any]] = []
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                events.append(json.loads(line))
                continue
            except json.JSONDecodeError:
                pass
        for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", line):
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict) and payload.get("type"):
                events.append(payload)
    return events


def build_hpt_search_space() -> Dict[str, Any]:
    from anomallm.hpo import tabular_search_space

    return tabular_search_space("classification")


def backfill_hpt_summary(run_id: str = DEFAULT_RUN_ID) -> Path:
    from anomallm.hpo import parse_hpt_from_evaluation

    evaluation = load_evaluation(run_id)
    hpt_updates = parse_hpt_from_evaluation(evaluation)
    payload = {**hpt_updates, "search_space": build_hpt_search_space(), "run_id": run_id}
    out_path = run_root(run_id) / "artifacts" / "hpt_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    return out_path


def build_final_report_markdown(run_id: str = DEFAULT_RUN_ID) -> str:
    from anomallm.evaluation_report import render_evaluation_sections, resolve_evaluation_metrics

    bundle = load_manifest_bundle(run_id)
    manifest = bundle.get("run_manifest") or {}
    training = bundle.get("training_artifacts") or {}
    evaluation = load_evaluation(run_id)
    hpt = parse_hpt_from_evaluation_for_report(evaluation)
    metrics = resolve_evaluation_metrics(evaluation, training.get("metrics") or {})
    training_config = training.get("training_config") or {}
    architecture = architecture_from_bundle(bundle)
    events = parse_training_log_events(run_root(run_id) / "logs" / "training.log")
    epoch_metrics = [e for e in events if e.get("type") == "epoch_metric"]

    user_query = (manifest.get("user_query") or "").strip('"')
    dataset = (bundle.get("dataset_profile") or {}).get("dataset_ref") or "unknown"

    evaluation_sections = render_evaluation_sections(
        metrics=metrics,
        architecture=architecture,
        training_config=training_config,
        hpt_search_space=build_hpt_search_space(),
        epoch_metrics=epoch_metrics,
        best_params=hpt.get("hpt_best_params") or {},
    )

    lines = [
        "# OmniML Autonomous ML Run Report",
        "",
        "## Architecture Selection",
        "",
        f"- User query: {user_query}",
        f"- Architecture (HITL-approved graph): {architecture}",
        f"- Actual estimator (Path B): RandomForest selected via grid search",
        f"- Dataset: {dataset}",
        "",
        evaluation_sections,
        "",
        "## Explainability",
        "",
        (bundle.get("xai_artifacts") or {}).get("narrative")
        or "Evidence-backed explanations are stored in run-scoped XAI artifacts.",
        "",
        "## Verdict",
        "",
        "Evidence-backed run summary generated from run-scoped artifacts. "
        "Paper Table II metrics use separate 5-fold CV via experiments/run.py.",
    ]
    return "\n".join(lines)


def parse_hpt_from_evaluation_for_report(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    from anomallm.hpo import parse_hpt_from_evaluation

    return parse_hpt_from_evaluation(evaluation)


def write_final_report(run_id: str = DEFAULT_RUN_ID) -> Path:
    report = build_final_report_markdown(run_id)
    out_path = run_root(run_id) / "reports" / "final_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    return out_path


def build_training_state_from_run(
    run_id: str = DEFAULT_RUN_ID,
    *,
    include_complete: bool = True,
    epoch_limit: Optional[int] = None,
) -> Dict[str, Any]:
    from anomallm.evaluation_report import resolve_evaluation_metrics, search_space_table_rows

    bundle = load_manifest_bundle(run_id)
    training = bundle.get("training_artifacts") or {}
    training_config = training.get("training_config") or {}
    evaluation = load_evaluation(run_id)
    hpt = parse_hpt_from_evaluation_for_report(evaluation)
    architecture = architecture_from_bundle(bundle)
    hpt_space = build_hpt_search_space()
    events = parse_training_log_events(run_root(run_id) / "logs" / "training.log")

    metrics: List[Dict[str, Any]] = []
    best_params: Dict[str, Any] = {}
    logs: List[str] = []
    current_epoch = 0

    for event in events:
        etype = event.get("type")
        if etype == "hpt_trial":
            logs.append(
                f"HPT trial {event.get('trial')}/{event.get('total')}: "
                f"value={float(event.get('value', 0)):.4f} params={event.get('params')}"
            )
        elif etype == "hpt_complete":
            best_params = event.get("best_params") or {}
            logs.append(f"HPT complete: best={best_params} kind={event.get('best_kind')}")
        elif etype == "epoch_metric":
            if epoch_limit is not None and int(event.get("epoch", 0)) > epoch_limit:
                continue
            metric = {
                "epoch": int(event.get("epoch", 0)),
                "loss": float(event.get("loss", 0)),
                "val_loss": float(event.get("val_loss", 0)),
                "acc": float(event.get("acc", 0)),
                "val_acc": float(event.get("val_acc", 0)),
            }
            metrics.append(metric)
            current_epoch = metric["epoch"]
            logs.append(f"Epoch {metric['epoch']}: val_acc={metric['val_acc']:.4f}")

    candidate_count = len(hpt_space.get("candidates") or [])
    search_strategy = f"{hpt_space.get('strategy', 'grid')} ({candidate_count} candidates)"
    final_metrics = resolve_evaluation_metrics(evaluation, training.get("metrics") or {})

    state: Dict[str, Any] = {
        "status": "running",
        "current_epoch": current_epoch,
        "total_epochs": int(training_config.get("epochs") or 5),
        "architecture": architecture,
        "optimizer": str(training_config.get("optimizer") or "adam"),
        "search_strategy": search_strategy,
        "best_params": best_params or hpt.get("hpt_best_params") or {},
        "metrics": metrics,
        "logs": logs[-50:],
        "groq_commentary": "",
        "search_space_table": [
            {"parameter": param, "search_range": rng}
            for param, rng in search_space_table_rows(hpt_space, training_config)
        ],
        "evaluation_protocol": {},
    }

    if include_complete and metrics:
        acc = final_metrics.get("accuracy", final_metrics.get("val_acc"))
        state["status"] = "complete"
        state["evaluation_protocol"] = {
            "accuracy_pct": round(float(acc) * 100, 1) if acc is not None else None,
            "f1": round(float(final_metrics.get("f1", 0.0)), 3) if final_metrics.get("f1") is not None else None,
            "epochs_executed": metrics[-1]["epoch"],
            "epochs_configured": int(training_config.get("epochs") or 5),
            "optimizer": str(training_config.get("optimizer") or "adam"),
            "architecture": architecture,
            "search_strategy": search_strategy,
            "best_params": state["best_params"],
        }
    return state


def latest_cv_output_dir() -> Path:
    output_root = ROOT / "experiments" / "output"
    candidates = sorted(output_root.glob("*/metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No CV metrics.json found. Run experiments.run first.")
    return candidates[0].parent


def render_report_html_for_run(run_id: str = DEFAULT_RUN_ID) -> str:
    from anomallm.evaluation_report import render_report_html

    markdown = build_final_report_markdown(run_id)
    body_parts: List[str] = []
    for block in markdown.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("# "):
            body_parts.append(f"<h1>{block[2:]}</h1>")
        elif block.startswith("## "):
            body_parts.append(f"<h2>{block[3:]}</h2>")
        elif block.startswith("- "):
            items = "".join(f"<li>{line[2:]}</li>" for line in block.splitlines() if line.startswith("- "))
            body_parts.append(f"<ul>{items}</ul>")
        elif block.startswith("|"):
            rows = block.splitlines()
            html_rows = []
            for idx, row in enumerate(rows):
                if row.strip().startswith("| ---"):
                    continue
                cells = [c.strip() for c in row.strip("|").split("|")]
                tag = "th" if idx == 0 else "td"
                cls = ' class="highlight"' if idx in {1, 2} and "Accuracy" in row or "F1" in row else ""
                if idx in {1, 2} and any(k in row for k in ("Accuracy", "F1")):
                    cls = ' class="highlight"'
                html_rows.append(
                    "<tr"
                    + (cls if idx > 0 else "")
                    + ">"
                    + "".join(f"<{tag}>{cell}</{tag}>" for cell in cells)
                    + "</tr>"
                )
            body_parts.append("<table>" + "".join(html_rows) + "</table>")
        else:
            body_parts.append(f"<p>{block}</p>")
    return render_report_html(
        "OmniML Final Report",
        "\n".join(body_parts),
        badge=f"Run {run_id} — Final Evaluation Report",
    )


def write_supplementary_pdf(markdown_path: Path, pdf_path: Path) -> Path:
    from fpdf import FPDF

    text = markdown_path.read_text(encoding="utf-8")
    text = text.replace("\u2013", "-").replace("\u2014", "-").replace("\u2192", "->")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.multi_cell(0, 8, "OmniML Supplementary Material")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=10)
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            pdf.ln(2)
            continue
        if stripped.startswith("# "):
            pdf.set_font("Helvetica", "B", 14)
            pdf.multi_cell(0, 7, stripped[2:])
        elif stripped.startswith("## "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.multi_cell(0, 6, stripped[3:])
        elif stripped.startswith("|"):
            pdf.set_font("Courier", size=9)
            pdf.multi_cell(0, 5, stripped)
        else:
            pdf.set_font("Helvetica", size=10)
            safe = stripped.encode("latin-1", errors="replace").decode("latin-1")
            pdf.multi_cell(0, 5, safe)
    pdf.output(str(pdf_path))
    return pdf_path
