from __future__ import annotations

import os
from typing import Dict, Iterable, List

from fpdf import FPDF

from .runtime import register_artifact, write_text
from .schemas import (
    ComplianceEvidenceRequirement,
    ComplianceReport,
    ComplianceSection,
    ComplianceTemplateId,
    ComplianceValidationResult,
    EvidenceBundle,
)


TEMPLATE_REQUIREMENTS: Dict[ComplianceTemplateId, List[ComplianceEvidenceRequirement]] = {
    "eu_ai_act": [
        ComplianceEvidenceRequirement(key="intended_purpose", description="Intended purpose narrative"),
        ComplianceEvidenceRequirement(key="dataset_provenance", description="Dataset provenance"),
        ComplianceEvidenceRequirement(key="training_metrics", description="Training metrics"),
        ComplianceEvidenceRequirement(key="xai", description="Transparency and explainability summary"),
        ComplianceEvidenceRequirement(key="human_oversight", description="Human oversight checkpoints"),
    ],
    "fda_samd": [
        ComplianceEvidenceRequirement(key="clinical_intended_use", description="Clinical intended use narrative"),
        ComplianceEvidenceRequirement(key="dataset_provenance", description="Training and validation provenance"),
        ComplianceEvidenceRequirement(key="performance", description="Performance evidence"),
        ComplianceEvidenceRequirement(key="fairness", description="Subgroup analysis when available", required=False, severity="warning"),
        ComplianceEvidenceRequirement(key="change_management", description="Version traceability"),
    ],
    "soc2": [
        ComplianceEvidenceRequirement(key="system_description", description="System description"),
        ComplianceEvidenceRequirement(key="change_management", description="Change management evidence"),
        ComplianceEvidenceRequirement(key="data_lineage", description="Data lineage"),
        ComplianceEvidenceRequirement(key="monitoring", description="Monitoring and logging"),
        ComplianceEvidenceRequirement(key="model_risk", description="Model risk controls"),
    ],
}


def _bundle_evidence_map(bundle: EvidenceBundle) -> Dict[str, str]:
    training = bundle.training_artifacts
    fairness = bundle.fairness_artifacts
    xai = bundle.xai_artifacts
    dataset = bundle.dataset_profile
    manifest = bundle.run_manifest
    plugin_evidence = [
        execution.details.get("evidence_block")
        for execution in bundle.plugin_artifacts.executions
        if execution.details.get("evidence_block")
    ]
    return {
        "intended_purpose": manifest.user_query,
        "clinical_intended_use": manifest.user_query,
        "dataset_provenance": dataset.provenance.get("summary") or dataset.dataset_ref or dataset.csv_path,
        "training_metrics": str(training.metrics),
        "performance": str(training.metrics),
        "xai": xai.narrative or str(xai.top_features),
        "fairness": fairness.narrative,
        "human_oversight": "Dataset selection, architecture review, training config approval, and execution mode selection are explicit HITL checkpoints.",
        "change_management": f"Run ID {manifest.run_id}, problem ID {manifest.problem_id}, evidence version {manifest.evidence_version}.",
        "system_description": "OmniML is a LangGraph-orchestrated ML system that sources data, trains a model, evaluates it, and renders evidence-backed reports.",
        "data_lineage": dataset.csv_path or dataset.dataset_ref,
        "monitoring": training.training_logs_path or "Training logs captured in run-scoped artifacts.",
        "model_risk": "Known limitations and failure modes are captured in the compliance report and XAI/fairness appendices.",
        "plugin_evidence": "\n".join(f"- {item.get('title')}: {item.get('body')}" for item in plugin_evidence) if plugin_evidence else "",
    }


def build_validation(template_id: ComplianceTemplateId, evidence_map: Dict[str, str]) -> ComplianceValidationResult:
    missing_required: List[str] = []
    warnings: List[str] = []
    assumptions: List[str] = []
    for requirement in TEMPLATE_REQUIREMENTS[template_id]:
        present = bool(evidence_map.get(requirement.key))
        if present:
            continue
        if requirement.required:
            missing_required.append(requirement.key)
        else:
            warnings.append(requirement.key)
            assumptions.append(f"{requirement.description} was not available and is explicitly marked as omitted.")
    completeness = "complete"
    if missing_required:
        completeness = "insufficient_evidence"
    elif warnings or assumptions:
        completeness = "complete_with_assumptions"
    return ComplianceValidationResult(
        completeness=completeness,
        missing_required=missing_required,
        assumptions=assumptions,
        warnings=warnings,
    )


def build_sections(template_id: ComplianceTemplateId, bundle: EvidenceBundle, validation: ComplianceValidationResult) -> List[ComplianceSection]:
    evidence_map = _bundle_evidence_map(bundle)
    sections: List[ComplianceSection] = []
    for requirement in TEMPLATE_REQUIREMENTS[template_id]:
        body = evidence_map.get(requirement.key) or "Evidence not available. This omission is explicitly disclosed."
        missing = [] if evidence_map.get(requirement.key) else [requirement.key]
        sections.append(
            ComplianceSection(
                title=requirement.description,
                key=requirement.key,
                body=body,
                evidence_keys=[requirement.key],
                missing_evidence=missing,
            )
        )
    sections.append(
        ComplianceSection(
            title="Evidence Traceability",
            key="evidence_traceability",
            body="\n".join(f"- {ref.name}: {ref.path}" for ref in bundle.run_manifest.artifact_refs) or "No artifact refs recorded.",
            evidence_keys=["artifact_refs"],
        )
    )
    if evidence_map.get("plugin_evidence"):
        sections.append(
            ComplianceSection(
                title="Plugin-Contributed Evidence",
                key="plugin_evidence",
                body=evidence_map["plugin_evidence"],
                evidence_keys=["plugin_evidence"],
            )
        )
    return sections


def render_markdown(report: ComplianceReport) -> str:
    lines = [f"# {report.title}", "", f"Completeness: **{report.validation.completeness}**", ""]
    if report.validation.missing_required:
        lines.extend(["## Missing Required Evidence", ""])
        lines.extend(f"- {item}" for item in report.validation.missing_required)
        lines.append("")
    if report.validation.assumptions:
        lines.extend(["## Assumptions", ""])
        lines.extend(f"- {item}" for item in report.validation.assumptions)
        lines.append("")
    for section in report.sections:
        lines.append(f"## {section.title}")
        lines.append("")
        lines.append(section.body)
        lines.append("")
    return "\n".join(lines)


def render_html(report: ComplianceReport, markdown: str) -> str:
    escaped = markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (
        "<html><head><meta charset='utf-8'><title>{}</title>"
        "<style>body{{font-family:Arial,sans-serif;margin:40px;line-height:1.45}}"
        "h1,h2{{color:#16324f}}</style></head>"
        "<body><pre style='white-space:pre-wrap'>{}</pre></body></html>"
    ).format(report.title, escaped)


def render_pdf(report: ComplianceReport, markdown: str, target_path: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, report.title, ln=1)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 8, f"Completeness: {report.validation.completeness}", ln=1)
    pdf.ln(4)
    for line in markdown.splitlines():
        text = line.strip()
        if not text:
            pdf.ln(2)
            continue
        if text.startswith("# "):
            pdf.set_font("Helvetica", "B", 15)
            pdf.multi_cell(0, 8, text[2:])
        elif text.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(0, 7, text[3:])
        else:
            pdf.set_font("Helvetica", size=10)
            pdf.multi_cell(0, 6, text)
    pdf.output(target_path)
    return target_path


def generate_compliance_reports(bundle: EvidenceBundle, template_ids: Iterable[ComplianceTemplateId]) -> List[ComplianceReport]:
    manifest = bundle.run_manifest
    reports: List[ComplianceReport] = []
    evidence_map = _bundle_evidence_map(bundle)
    for template_id in template_ids:
        validation = build_validation(template_id, evidence_map)
        report = ComplianceReport(
            template_id=template_id,
            title=f"OmniML Compliance Report - {template_id.replace('_', ' ').upper()}",
            validation=validation,
            sections=build_sections(template_id, bundle, validation),
        )
        markdown = render_markdown(report)
        markdown_path = os.path.join(manifest.paths["reports"], f"{template_id}.md")
        html_path = os.path.join(manifest.paths["reports"], f"{template_id}.html")
        pdf_path = os.path.join(manifest.paths["reports"], f"{template_id}.pdf")
        write_text(markdown_path, markdown)
        write_text(html_path, render_html(report, markdown))
        render_pdf(report, markdown, pdf_path)
        report.markdown_path = markdown_path
        report.html_path = html_path
        report.pdf_path = pdf_path
        register_artifact(manifest, f"{template_id}_markdown", "report_markdown", markdown_path)
        register_artifact(manifest, f"{template_id}_html", "report_html", html_path)
        register_artifact(manifest, f"{template_id}_pdf", "report_pdf", pdf_path)
        reports.append(report)
    return reports
