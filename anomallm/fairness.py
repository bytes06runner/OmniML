from __future__ import annotations

import csv
import os
from typing import Any, Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .runtime import write_json
from .schemas import FairnessAuditResult, FairnessConfig, FairnessFinding, SensitiveFeatureSpec, SubgroupMetricRow


SENSITIVE_HINTS = (
    "gender",
    "sex",
    "race",
    "ethnicity",
    "age",
    "religion",
    "disability",
    "marital",
    "nationality",
)


def detect_sensitive_features(columns: List[str]) -> List[str]:
    detected = []
    for col in columns:
        lowered = col.lower()
        if any(hint in lowered for hint in SENSITIVE_HINTS):
            detected.append(col)
    return detected


def _classification_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def _build_fairness_narrative(status: str, confirmed: List[str], findings: List[FairnessFinding]) -> str:
    if status == "not_applicable":
        return "No demographic or protected attributes were available, so fairness analysis was not applicable for this run."
    if status == "insufficient_subgroup_support":
        return "Candidate demographic attributes were detected, but subgroup counts were too small for a reliable fairness conclusion."
    if findings:
        joined = "; ".join(f"{finding.feature}:{finding.metric} disparity={finding.disparity:.4f}" for finding in findings)
        return f"Fairness audit identified material disparities across protected attributes. Findings: {joined}."
    if confirmed:
        return "Fairness audit completed for the selected protected attributes without material threshold breaches."
    return "Fairness analysis could not be completed because required prediction evidence was unavailable."


def run_fairness_audit(
    dataset_path: str,
    predictions_path: str,
    run_artifacts_dir: str,
    config: Dict[str, Any] | None = None,
) -> FairnessAuditResult:
    fairness_config = FairnessConfig.model_validate(config or {})
    if not dataset_path or not os.path.exists(dataset_path) or not predictions_path or not os.path.exists(predictions_path):
        return FairnessAuditResult(status="not_available", config=fairness_config)

    dataset = pd.read_csv(dataset_path)
    predictions = pd.read_csv(predictions_path)
    if "y_true" not in predictions.columns or "y_pred" not in predictions.columns:
        return FairnessAuditResult(status="not_available", config=fairness_config)

    detected = detect_sensitive_features(dataset.columns.tolist())
    confirmed = fairness_config.protected_attributes or detected
    specs = [SensitiveFeatureSpec(name=name, source="heuristic", confirmed=name in confirmed) for name in detected]
    if not confirmed:
        return FairnessAuditResult(status="not_applicable", config=fairness_config, detected_sensitive_features=specs)

    merged = dataset.iloc[: len(predictions)].copy()
    merged["y_true"] = predictions["y_true"].values[: len(merged)]
    merged["y_pred"] = predictions["y_pred"].values[: len(merged)]

    group_rows: List[SubgroupMetricRow] = []
    findings: List[FairnessFinding] = []
    insufficient = False
    for feature in confirmed:
        if feature not in merged.columns:
            continue
        for group, group_df in merged.groupby(feature):
            if len(group_df) < fairness_config.minimum_group_size:
                insufficient = True
                continue
            metrics = _classification_metrics(group_df["y_true"], group_df["y_pred"])
            group_rows.append(
                SubgroupMetricRow(feature=feature, group=str(group), sample_size=len(group_df), metrics=metrics)
            )

        rows_for_feature = [row for row in group_rows if row.feature == feature]
        values = [row.metrics.get(fairness_config.primary_metric, 0.0) for row in rows_for_feature]
        groups = [row.group for row in rows_for_feature]
        if len(values) >= 2:
            disparity = max(values) - min(values)
            if disparity > fairness_config.disparity_threshold:
                findings.append(
                    FairnessFinding(
                        feature=feature,
                        metric=fairness_config.primary_metric,
                        groups=groups,
                        disparity=round(disparity, 6),
                        threshold=fairness_config.disparity_threshold,
                        rationale=(
                            f"{fairness_config.primary_metric} varies by {disparity:.4f} across {feature} groups, "
                            "which exceeds the configured disparity threshold."
                        ),
                    )
                )

    os.makedirs(run_artifacts_dir, exist_ok=True)
    summary_path = os.path.join(run_artifacts_dir, "fairness_summary.json")
    group_metrics_path = os.path.join(run_artifacts_dir, "group_metrics.csv")
    with open(group_metrics_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "group", "sample_size", "accuracy", "precision", "recall", "f1"])
        for row in group_rows:
            writer.writerow([
                row.feature,
                row.group,
                row.sample_size,
                row.metrics.get("accuracy"),
                row.metrics.get("precision"),
                row.metrics.get("recall"),
                row.metrics.get("f1"),
            ])

    if findings:
        status = "evaluated_with_findings"
    elif group_rows:
        status = "evaluated_no_material_findings"
    elif insufficient:
        status = "insufficient_subgroup_support"
    else:
        status = "not_available"

    result = FairnessAuditResult(
        status=status,
        config=fairness_config,
        detected_sensitive_features=specs,
        confirmed_sensitive_features=confirmed,
        group_metrics=group_rows,
        findings=findings,
        summary_path=summary_path,
        group_metrics_path=group_metrics_path,
        narrative=_build_fairness_narrative(status, confirmed, findings),
    )
    write_json(summary_path, result.model_dump(mode="json"))
    return result
