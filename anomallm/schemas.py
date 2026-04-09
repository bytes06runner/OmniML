from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ComplianceTemplateId = Literal["eu_ai_act", "fda_samd", "soc2"]
ComplianceCompleteness = Literal["complete", "complete_with_assumptions", "insufficient_evidence"]
FairnessStatus = Literal[
    "not_applicable",
    "not_available",
    "insufficient_subgroup_support",
    "evaluated_with_findings",
    "evaluated_no_material_findings",
]


class ArtifactRef(BaseModel):
    name: str
    kind: str
    path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RunManifest(BaseModel):
    run_id: str
    user_query: str
    problem_id: str
    status: str = "running"
    evidence_version: str = "1.0"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    paths: Dict[str, str] = Field(default_factory=dict)
    artifact_refs: List[ArtifactRef] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DatasetProfile(BaseModel):
    source: str = ""
    dataset_ref: str = ""
    csv_path: str = ""
    row_count: int = 0
    col_count: int = 0
    columns: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    modality: str = "tabular"
    provenance: Dict[str, Any] = Field(default_factory=dict)
    lineage: Dict[str, Any] = Field(default_factory=dict)
    missing: Dict[str, Any] = Field(default_factory=dict)
    top_correlations: List[Dict[str, Any]] = Field(default_factory=list)
    detected_sensitive_features: List[str] = Field(default_factory=list)


class TrainingArtifacts(BaseModel):
    task_type: str = "unknown"
    metrics: Dict[str, Any] = Field(default_factory=dict)
    best_params: Dict[str, Any] = Field(default_factory=dict)
    training_config: Dict[str, Any] = Field(default_factory=dict)
    training_logs_path: Optional[str] = None
    predictions_path: Optional[str] = None
    evaluation_path: Optional[str] = None
    model_card: Dict[str, Any] = Field(default_factory=dict)
    export_paths: Dict[str, str] = Field(default_factory=dict)
    plots: Dict[str, str] = Field(default_factory=dict)


class XAIArtifacts(BaseModel):
    status: str = "not_generated"
    explanation_method: str = "heuristic"
    narrative: str = ""
    top_features: List[Dict[str, Any]] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    evidence: Dict[str, Any] = Field(default_factory=dict)


class FairnessConfig(BaseModel):
    protected_attributes: List[str] = Field(default_factory=list)
    primary_metric: str = "accuracy"
    disparity_threshold: float = 0.05
    minimum_group_size: int = 25
    backend: Literal["fairlearn", "aif360"] = "fairlearn"


class SensitiveFeatureSpec(BaseModel):
    name: str
    source: str = "heuristic"
    confirmed: bool = False
    excluded: bool = False


class SubgroupMetricRow(BaseModel):
    feature: str
    group: str
    sample_size: int
    metrics: Dict[str, float] = Field(default_factory=dict)


class FairnessFinding(BaseModel):
    feature: str
    metric: str
    groups: List[str] = Field(default_factory=list)
    disparity: float
    threshold: float
    severity: str = "warning"
    rationale: str


class FairnessAuditResult(BaseModel):
    status: FairnessStatus = "not_available"
    config: FairnessConfig = Field(default_factory=FairnessConfig)
    detected_sensitive_features: List[SensitiveFeatureSpec] = Field(default_factory=list)
    confirmed_sensitive_features: List[str] = Field(default_factory=list)
    excluded_sensitive_features: List[str] = Field(default_factory=list)
    group_metrics: List[SubgroupMetricRow] = Field(default_factory=list)
    findings: List[FairnessFinding] = Field(default_factory=list)
    narrative: str = ""
    summary_path: Optional[str] = None
    group_metrics_path: Optional[str] = None
    chart_paths: Dict[str, str] = Field(default_factory=dict)


class BenchmarkQuery(BaseModel):
    raw_prompt: str
    task_label: str
    modality: str = "tabular"
    metric_name: str = "accuracy"


class LeaderboardEntry(BaseModel):
    source: str
    title: str
    dataset: str = ""
    metric_name: str = ""
    metric_value: Optional[float] = None
    value_text: str = ""
    rank: Optional[int] = None
    url: str = ""
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)


class ComparabilityAssessment(BaseModel):
    score: float = 0.0
    directly_comparable: bool = False
    notes: List[str] = Field(default_factory=list)
    matched_dimensions: Dict[str, bool] = Field(default_factory=dict)


class GapRecommendation(BaseModel):
    title: str
    category: Literal["evidence_backed", "heuristic"] = "heuristic"
    recommendation: str
    source_refs: List[str] = Field(default_factory=list)


class BenchmarkArtifacts(BaseModel):
    query: Optional[BenchmarkQuery] = None
    leaderboard_entries: List[LeaderboardEntry] = Field(default_factory=list)
    comparability: ComparabilityAssessment = Field(default_factory=ComparabilityAssessment)
    cited_papers: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[GapRecommendation] = Field(default_factory=list)
    narrative: str = ""
    rendered_markdown: str = ""
    source_status: Literal["live", "cached_fresh", "cached_stale", "unavailable"] = "unavailable"
    retrieval_failures: List[str] = Field(default_factory=list)
    cache_hit: bool = False
    retrieved_at: Optional[datetime] = None
    staleness_days: Optional[int] = None


class ComplianceEvidenceRequirement(BaseModel):
    key: str
    description: str
    required: bool = True
    severity: Literal["error", "warning"] = "error"


class ComplianceSection(BaseModel):
    title: str
    key: str
    body: str
    evidence_keys: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list)


class ComplianceValidationResult(BaseModel):
    completeness: ComplianceCompleteness = "insufficient_evidence"
    missing_required: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    template_id: ComplianceTemplateId
    title: str
    validation: ComplianceValidationResult
    sections: List[ComplianceSection] = Field(default_factory=list)
    markdown_path: Optional[str] = None
    html_path: Optional[str] = None
    pdf_path: Optional[str] = None


class PluginManifest(BaseModel):
    name: str
    version: str
    module: str
    class_name: str
    slot: str
    description: str = ""
    api_version: str = "1.0"
    task_types: List[str] = Field(default_factory=list)
    compliance_impact: str = ""
    config_schema: Dict[str, Any] = Field(default_factory=dict)


class PluginExecutionRecord(BaseModel):
    plugin_name: str
    slot: str
    status: str
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class PluginArtifacts(BaseModel):
    discovered: List[PluginManifest] = Field(default_factory=list)
    executions: List[PluginExecutionRecord] = Field(default_factory=list)
    enabled_plugins: List[str] = Field(default_factory=list)
    catalog: List[PluginManifest] = Field(default_factory=list)


class EvidenceBundle(BaseModel):
    run_manifest: RunManifest
    dataset_profile: DatasetProfile = Field(default_factory=DatasetProfile)
    training_artifacts: TrainingArtifacts = Field(default_factory=TrainingArtifacts)
    xai_artifacts: XAIArtifacts = Field(default_factory=XAIArtifacts)
    fairness_artifacts: FairnessAuditResult = Field(default_factory=FairnessAuditResult)
    benchmark_artifacts: BenchmarkArtifacts = Field(default_factory=BenchmarkArtifacts)
    compliance_artifacts: List[ComplianceReport] = Field(default_factory=list)
    plugin_artifacts: PluginArtifacts = Field(default_factory=PluginArtifacts)
