from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from .persistence import SQLiteDataLayer
from .schemas import BenchmarkArtifacts, BenchmarkQuery, ComparabilityAssessment, GapRecommendation, LeaderboardEntry


TASK_HINTS = {
    "fraud": "fraud_detection",
    "breast cancer": "medical_diagnosis_classification",
    "diagnose": "medical_diagnosis_classification",
    "churn": "churn_prediction",
    "house price": "tabular_regression",
    "forecast": "time_series_forecasting",
}


def normalize_task_label(user_query: str, dataset_ref: str = "", modality: str = "tabular") -> BenchmarkQuery:
    lowered = f"{user_query} {dataset_ref}".lower()
    task_label = "tabular_classification"
    for hint, label in TASK_HINTS.items():
        if hint in lowered:
            task_label = label
            break
    metric_name = "accuracy"
    if "regression" in task_label or "price" in lowered:
        metric_name = "rmse"
    return BenchmarkQuery(raw_prompt=user_query, task_label=task_label, modality=modality, metric_name=metric_name)


def parse_arxiv_results(raw: str) -> List[Dict[str, str]]:
    papers: List[Dict[str, str]] = []
    blocks = [block.strip() for block in raw.split("\n---\n") if block.strip()]
    for block in blocks:
        paper: Dict[str, str] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            paper[key.strip().lower()] = value.strip()
        if paper:
            papers.append(paper)
    return papers


def assess_comparability(query: BenchmarkQuery, metrics: Dict[str, Any], leaderboard_entries: List[LeaderboardEntry], dataset_ref: str) -> ComparabilityAssessment:
    score = 0.0
    matched = {"task": True, "metric": False, "dataset": False, "modality": query.modality == "tabular"}
    notes: List[str] = [f"Normalized benchmark task: {query.task_label}."]
    if metrics and query.metric_name in ("accuracy", "rmse"):
        score += 0.3
        matched["metric"] = True
    if dataset_ref:
        lowered = dataset_ref.lower()
        for entry in leaderboard_entries:
            if entry.dataset and entry.dataset.lower() in lowered:
                matched["dataset"] = True
                score += 0.3
                notes.append(f"Dataset appears aligned with leaderboard entry '{entry.dataset}'.")
                break
    if leaderboard_entries:
        score += 0.2
        notes.append(f"Retrieved {len(leaderboard_entries)} leaderboard candidates.")
    if matched["modality"]:
        score += 0.2
    directly_comparable = score >= 0.6
    if not directly_comparable:
        notes.append("Benchmark sources are related but not strongly protocol-matched to the current run.")
    return ComparabilityAssessment(score=round(score, 3), directly_comparable=directly_comparable, notes=notes, matched_dimensions=matched)


def build_gap_recommendations(comparability: ComparabilityAssessment, papers: List[Dict[str, str]], task_label: str) -> List[GapRecommendation]:
    recommendations: List[GapRecommendation] = []
    if papers:
        recommendations.append(
            GapRecommendation(
                title="Review recent literature features",
                category="evidence_backed",
                recommendation=f"Inspect the latest retrieved {task_label} papers for architectural motifs and preprocessing choices before altering the model.",
                source_refs=[paper.get("link", "") for paper in papers[:3] if paper.get("link")],
            )
        )
    recommendations.append(
        GapRecommendation(
            title="Tighten comparability",
            category="heuristic",
            recommendation="Match the benchmark dataset split and evaluation metric exactly before treating a benchmark gap as actionably real.",
        )
    )
    if comparability.directly_comparable:
        recommendations.append(
            GapRecommendation(
                title="Close the measured gap",
                category="heuristic",
                recommendation="Investigate feature engineering, stronger regularization, and a larger hyperparameter search budget to reduce the measured benchmark delta.",
            )
        )
    return recommendations


def render_benchmark_markdown(artifacts: BenchmarkArtifacts) -> str:
    lines = [
        "## Benchmark Gap Analysis",
        "",
        f"- Task label: `{artifacts.query.task_label if artifacts.query else 'unknown'}`",
        f"- Source status: `{artifacts.source_status}`",
        f"- Cache hit: `{artifacts.cache_hit}`",
        f"- Retrieved at: `{artifacts.retrieved_at.isoformat() if artifacts.retrieved_at else 'n/a'}`",
        f"- Comparability score: `{artifacts.comparability.score}`",
        f"- Directly comparable: `{artifacts.comparability.directly_comparable}`",
        "",
        "### Sources",
    ]
    if artifacts.leaderboard_entries:
        for entry in artifacts.leaderboard_entries[:5]:
            lines.append(
                f"- {entry.title} | source={entry.source} | dataset={entry.dataset or 'n/a'} | "
                f"url={entry.url or 'n/a'} | retrieved={entry.retrieved_at.date().isoformat()}"
            )
    else:
        lines.append("- No leaderboard entries were available; related literature context only.")
    if artifacts.retrieval_failures:
        lines.extend(["", "### Retrieval Failures"])
        lines.extend(f"- {failure}" for failure in artifacts.retrieval_failures)
    lines.extend(["", "### Recommendations"])
    for rec in artifacts.recommendations:
        lines.append(f"- [{rec.category}] {rec.recommendation}")
    return "\n".join(lines)


class PapersWithCodeSource:
    source_type = "papers_with_code"

    def __init__(self, timeout_seconds: int = 10):
        self.timeout_seconds = timeout_seconds

    def fetch(self, query: BenchmarkQuery) -> tuple[list[LeaderboardEntry], str]:
        entries: List[LeaderboardEntry] = []
        response = requests.get(
            "https://paperswithcode.com/api/v1/search/",
            params={"q": query.task_label, "items_per_page": 5},
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        excerpt = str(payload)[:1500]
        for index, item in enumerate(payload.get("results", [])[:5], start=1):
            entries.append(
                LeaderboardEntry(
                    source=self.source_type,
                    title=item.get("name") or item.get("paper_title") or query.task_label,
                    dataset=item.get("dataset", "") or "",
                    metric_name=query.metric_name,
                    value_text=item.get("description", "")[:160],
                    rank=index,
                    url=item.get("url_abs") or item.get("url") or "",
                    retrieved_at=datetime.now(timezone.utc),
                )
            )
        return entries, excerpt


class ArxivContextSource:
    source_type = "arxiv"

    def parse(self, raw: str) -> List[Dict[str, str]]:
        return parse_arxiv_results(raw)


@dataclass
class BenchmarkCache:
    db: SQLiteDataLayer

    def load(self, task_label: str) -> Optional[Dict[str, Any]]:
        return self.db.get_latest_benchmark_cache(task_label)

    def save(self, task_label: str, source_status: str, source_type: str, payload: Dict[str, Any], raw_excerpt: str = "", failure_reason: str = "") -> None:
        self.db.save_benchmark_cache(task_label, source_status, source_type, payload, raw_excerpt=raw_excerpt, failure_reason=failure_reason)


class BenchmarkSourceManager:
    def __init__(self, cache: Optional[BenchmarkCache] = None, timeout_seconds: int = 10, cache_ttl_days: int = 7):
        self.cache = cache or BenchmarkCache(SQLiteDataLayer())
        self.pwc = PapersWithCodeSource(timeout_seconds=timeout_seconds)
        self.arxiv = ArxivContextSource()
        self.cache_ttl_days = cache_ttl_days

    def resolve(
        self,
        query: BenchmarkQuery,
        literature_raw: str,
        mode: str = "prefer_live_then_cache",
    ) -> BenchmarkArtifacts:
        now = datetime.now(timezone.utc)
        failures: List[str] = []
        leaderboard_entries: List[LeaderboardEntry] = []
        cache_hit = False
        source_status = "unavailable"
        raw_excerpt = ""

        if mode != "cache_only":
            try:
                leaderboard_entries, raw_excerpt = self.pwc.fetch(query)
                source_status = "live" if leaderboard_entries else "unavailable"
                self.cache.save(
                    query.task_label,
                    source_status=source_status,
                    source_type=self.pwc.source_type,
                    payload={"leaderboard_entries": [entry.model_dump(mode="json") for entry in leaderboard_entries]},
                    raw_excerpt=raw_excerpt,
                )
            except Exception as exc:
                failures.append(f"Papers With Code retrieval failed: {exc}")

        if (not leaderboard_entries and mode != "live_only") or mode == "cache_only":
            cached = self.cache.load(query.task_label)
            if cached:
                cache_hit = True
                cached_entries = cached.get("payload", {}).get("leaderboard_entries", [])
                leaderboard_entries = [LeaderboardEntry.model_validate(entry) for entry in cached_entries]
                retrieved = datetime.fromisoformat(cached["retrieved_at"]) if cached.get("retrieved_at") else now
                staleness_days = max((now - retrieved.replace(tzinfo=timezone.utc if retrieved.tzinfo is None else retrieved.tzinfo)).days, 0)
                source_status = "cached_stale" if staleness_days > self.cache_ttl_days else "cached_fresh"
            else:
                staleness_days = None
        else:
            staleness_days = 0

        papers = self.arxiv.parse(literature_raw) if literature_raw and not literature_raw.startswith("ERROR") else []
        artifacts = BenchmarkArtifacts(
            query=query,
            leaderboard_entries=leaderboard_entries,
            cited_papers=papers,
            source_status=source_status if leaderboard_entries else "unavailable",
            retrieval_failures=failures,
            cache_hit=cache_hit,
            retrieved_at=now,
            staleness_days=staleness_days,
        )
        return artifacts
