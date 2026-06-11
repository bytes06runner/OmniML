from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _fmt_float(value: float) -> str:
    if value >= 0.01 or value == 0.0:
        return f"{value:g}"
    return f"{value:.0e}"


def _format_range(values: List[Any]) -> str:
    if not values:
        return "N/A"
    if len(values) == 1:
        return str(values[0])
    if all(isinstance(v, (int, float)) for v in values):
        nums = sorted(float(v) for v in values)
        if nums[0] == nums[-1]:
            return _fmt_float(nums[0])
        return f"{_fmt_float(nums[0])}–{_fmt_float(nums[-1])}"
    return ", ".join(str(v) for v in values)


def _rows_from_optuna_style(space: Dict[str, Any]) -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for name, spec in space.items():
        if not isinstance(spec, (list, tuple)) or len(spec) < 2:
            continue
        kind = spec[0]
        if kind == "float_log" and len(spec) >= 3:
            rows.append((name.replace("_", " ").title(), f"{_fmt_float(float(spec[1]))}–{_fmt_float(float(spec[2]))} (log-uniform)"))
        elif kind == "float" and len(spec) >= 3:
            rows.append((name.replace("_", " ").title(), f"{float(spec[1]):g}–{float(spec[2]):g}"))
        elif kind == "categorical" and len(spec) >= 2:
            options = spec[1]
            if name == "optimizer":
                rows.append(("Optimizer", ", ".join(str(o).title() for o in options)))
            elif name == "batch_size":
                rows.append(("Batch Size", _format_range(list(options))))
            elif name == "learning_rate":
                rows.append(("Learning Rate", _format_range([float(o) for o in options])))
            elif name.startswith("units_"):
                rows.append(("Hidden Units", _format_range(list(options))))
            elif name.startswith("rate_"):
                rows.append(("Dropout", f"{min(options):g}–{max(options):g}"))
            else:
                rows.append((name.replace("_", " ").title(), _format_range(list(options))))
    return rows


def search_space_table_rows(
    hpt_search_space: Optional[Dict[str, Any]],
    training_config: Optional[Dict[str, Any]] = None,
) -> List[Tuple[str, str]]:
    """Return (parameter, search_range) rows for the active HPT space."""
    space = hpt_search_space or {}
    training_config = training_config or {}
    rows: List[Tuple[str, str]] = []

    graph_space = space.get("graph_space") or {}
    inner = graph_space.get("space") if isinstance(graph_space.get("space"), dict) else graph_space
    if isinstance(inner, dict) and inner:
        rows.extend(_rows_from_optuna_style(inner))

    candidates = space.get("candidates") or []
    if candidates:
        kinds = sorted({str(c.get("kind", "")) for c in candidates if c.get("kind")})
        if kinds:
            rows.append(("Model / Estimator", ", ".join(kinds)))
        param_values: Dict[str, List[Any]] = {}
        for candidate in candidates:
            params = candidate.get("params") or {}
            for key, value in params.items():
                param_values.setdefault(key, [])
                if value not in param_values[key]:
                    param_values[key].append(value)
        label_map = {
            "max_depth": "Max Depth",
            "n_estimators": "N Estimators",
            "C": "Regularization (C)",
            "learning_rate": "Learning Rate",
            "batch_size": "Batch Size",
        }
        for key in sorted(param_values):
            label = label_map.get(key, key.replace("_", " ").title())
            values = param_values[key]
            display = []
            for value in values:
                if value is None:
                    display.append("unlimited")
                else:
                    display.append(str(value))
            rows.append((label, ", ".join(display)))

    if not rows:
        rows = [
            ("Learning Rate", _format_range([float(training_config.get("lr") or 0.001)])),
            ("Batch Size", str(training_config.get("batch_size") or 64)),
            ("Optimizer", str(training_config.get("optimizer") or "adam").title()),
            ("Epochs", str(training_config.get("epochs") or 50)),
        ]

    seen = set()
    deduped: List[Tuple[str, str]] = []
    for param, rng in rows:
        key = param.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append((param, rng))
    return deduped


def render_search_space_table(
    hpt_search_space: Optional[Dict[str, Any]],
    training_config: Optional[Dict[str, Any]] = None,
) -> str:
    rows = search_space_table_rows(hpt_search_space, training_config)
    lines = [
        "## Hyperparameter Search Space",
        "",
        "| Parameter | Search Range |",
        "| --- | --- |",
    ]
    for param, rng in rows:
        lines.append(f"| {param} | {rng} |")
    return "\n".join(lines)


def resolve_evaluation_metrics(
    evaluation_payload: Optional[Dict[str, Any]],
    training_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    metrics = dict((evaluation_payload or {}).get("metrics") or {})
    if not metrics:
        metrics = dict(training_metrics or {})
    if metrics.get("accuracy") is None and metrics.get("val_acc") is not None:
        metrics["accuracy"] = metrics["val_acc"]
    return metrics


def resolve_epochs(
    training_config: Optional[Dict[str, Any]],
    epoch_metrics: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[int, int]:
    training_config = training_config or {}
    configured = int(training_config.get("epochs") or 50)
    executed = configured
    if epoch_metrics:
        executed = max(int(m.get("epoch") or 0) for m in epoch_metrics)
    elif training_config.get("epochs"):
        executed = int(training_config["epochs"])
    return configured, executed


def resolve_optimizer(training_config: Optional[Dict[str, Any]], hpt_search_space: Optional[Dict[str, Any]] = None) -> str:
    training_config = training_config or {}
    if training_config.get("optimizer"):
        return str(training_config["optimizer"]).title()
    space = hpt_search_space or {}
    graph_space = space.get("graph_space") or {}
    inner = graph_space.get("space") if isinstance(graph_space.get("space"), dict) else graph_space
    if isinstance(inner, dict):
        spec = inner.get("optimizer")
        if isinstance(spec, (list, tuple)) and len(spec) >= 2 and spec[0] == "categorical":
            return ", ".join(str(o).title() for o in spec[1])
    task_type = space.get("task_type")
    if task_type == "classification" or space.get("strategy") == "grid":
        return "N/A (sklearn grid: RF + LogReg)"
    return "Adam (default)"


def render_evaluation_protocol(
    *,
    metrics: Dict[str, Any],
    architecture: str,
    training_config: Optional[Dict[str, Any]] = None,
    hpt_search_space: Optional[Dict[str, Any]] = None,
    epoch_metrics: Optional[List[Dict[str, Any]]] = None,
    best_params: Optional[Dict[str, Any]] = None,
) -> str:
    training_config = training_config or {}
    configured_epochs, executed_epochs = resolve_epochs(training_config, epoch_metrics)
    optimizer = resolve_optimizer(training_config, hpt_search_space)
    strategy = (hpt_search_space or {}).get("strategy") or "grid"
    candidate_count = len((hpt_search_space or {}).get("candidates") or [])
    search_summary = f"{strategy} ({candidate_count} candidates)" if candidate_count else strategy

    accuracy = metrics.get("accuracy")
    if accuracy is None:
        accuracy = metrics.get("val_acc")
    f1 = metrics.get("f1")
    rmse = metrics.get("rmse")

    lines = [
        "## Evaluation Protocol",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    if accuracy is not None:
        lines.append(f"| Accuracy | {float(accuracy) * 100:.1f}% |")
    if f1 is not None:
        lines.append(f"| F1 | {float(f1):.3f} |")
    if rmse is not None:
        lines.append(f"| RMSE | {float(rmse):.4f} |")
    lines.extend(
        [
            f"| Epochs | {executed_epochs} executed / {configured_epochs} configured |",
            f"| Optimizer | {optimizer} |",
            f"| Architecture | {architecture or 'N/A'} |",
            f"| Search Strategy | {search_summary} |",
        ]
    )
    if best_params:
        lines.append(f"| Selected Hyperparameters | `{best_params}` |")
    return "\n".join(lines)


def render_evaluation_sections(
    *,
    metrics: Dict[str, Any],
    architecture: str,
    training_config: Optional[Dict[str, Any]] = None,
    hpt_search_space: Optional[Dict[str, Any]] = None,
    epoch_metrics: Optional[List[Dict[str, Any]]] = None,
    best_params: Optional[Dict[str, Any]] = None,
) -> str:
    protocol = render_evaluation_protocol(
        metrics=metrics,
        architecture=architecture,
        training_config=training_config,
        hpt_search_space=hpt_search_space,
        epoch_metrics=epoch_metrics,
        best_params=best_params,
    )
    table = render_search_space_table(hpt_search_space, training_config)
    return f"{protocol}\n\n{table}"


REPORT_HTML_STYLES = """
html {
  color-scheme: light only;
  background: #ffffff;
}
body {
  font-family: "Segoe UI", Arial, sans-serif;
  max-width: 920px;
  margin: 40px auto;
  padding: 0 24px 48px;
  line-height: 1.55;
  background: #ffffff !important;
  color: #111827 !important;
}
h1, h2, h3, p, li, td, th, code, span, div {
  color: inherit;
}
h1 {
  color: #111827 !important;
  border-bottom: 2px solid #6366f1;
  padding-bottom: 8px;
}
h2 {
  color: #1f2937 !important;
  margin-top: 28px;
}
p, li {
  color: #111827 !important;
}
table {
  border-collapse: collapse;
  width: 100%;
  margin: 12px 0 20px;
  background: #ffffff;
}
th, td {
  border: 1px solid #9ca3af;
  padding: 10px 12px;
  text-align: left;
  background: #ffffff !important;
  color: #111827 !important;
}
th {
  background: #e5e7eb !important;
  color: #111827 !important;
  font-weight: 700;
}
tbody tr:nth-child(even) td {
  background: #f9fafb !important;
}
tbody tr.highlight td {
  background: #d1fae5 !important;
  color: #065f46 !important;
  font-weight: 600;
}
code {
  background: #f3f4f6 !important;
  color: #111827 !important;
  padding: 2px 6px;
  border-radius: 4px;
}
.badge {
  display: inline-block;
  background: #6366f1 !important;
  color: #ffffff !important;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  margin-bottom: 16px;
}
pre {
  white-space: pre-wrap;
  background: #f9fafb !important;
  color: #111827 !important;
  border: 1px solid #d1d5db;
  padding: 16px;
  border-radius: 8px;
}
"""


def render_report_html(title: str, body_html: str, badge: str = "") -> str:
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    return (
        f"<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
        f"<meta name=\"color-scheme\" content=\"light only\">"
        f"<title>{title}</title><style>{REPORT_HTML_STYLES}</style></head>"
        f"<body>{badge_html}{body_html}</body></html>"
    )

