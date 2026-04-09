"""
graph.py — AnomaLLM v3 LangGraph Pipeline
==========================================
Architecture:
  Node 1: architect_sourcer    → Analyze query, search Kaggle, suggest architecture
  HITL Pause                   → User selects dataset via Chainlit UI
  Node 2: dataset_downloader   → Download the selected dataset CSV to disk (REAL)
  Node 3: engineer             → Generate full PyTorch training script using real CSV path
  Node 4: execution_sandbox    → Run script in subprocess, capture stdout/stderr
  Node 5: evaluator            → Produce a structured Markdown report

Author: AnomaLLM v3 / Antigravity
"""

import os
import re
import sys
import json
import uuid
import logging
import subprocess
import textwrap
import tempfile
import threading
from typing import TypedDict, List, Optional, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt
from anomallm.runtime import create_run_manifest, ensure_run_paths, load_or_create_bundle, persist_bundle, register_artifact, write_json, write_text
from anomallm.schemas import DatasetProfile, TrainingArtifacts, BenchmarkArtifacts, FairnessAuditResult, PluginArtifacts, XAIArtifacts
from anomallm.fairness import run_fairness_audit
from anomallm.compliance import generate_compliance_reports
from anomallm.benchmarking import (
    normalize_task_label,
    parse_arxiv_results,
    assess_comparability,
    build_gap_recommendations,
    render_benchmark_markdown,
    BenchmarkSourceManager,
)
from anomallm.plugins import PluginRegistry

# ── Safe Logger — prevents OSError [Errno 5] in async Chainlit context ───────
_logger = logging.getLogger("omniml.graph")
logging.basicConfig(level=logging.INFO, format="%(message)s")

def _safe_log(msg: str) -> None:
    """Print to stdout safely; if stdout is broken (Errno 5), fall back to logger."""
    try:
        print(msg, flush=True)
    except OSError:
        _logger.info(msg)

# ── Global EDA Progress Store ────────────────────────────────────────────────
# Thread-safe ring buffer; app.py SSE endpoint reads from this
_eda_lock   = threading.Lock()
_eda_steps  = {}   # session_id -> list of step dicts
_eda_done   = {}   # session_id -> bool

def _eda_emit(session_id: str, step_id: str, label: str, status: str = "running",
              detail: str = "", pct: int = 0, data: dict | None = None) -> None:
    """Push one progress event into the ring buffer (called from eda_analyzer_node)."""
    event = {
        "id": step_id, "label": label, "status": status,
        "detail": detail, "pct": pct, "data": data or {}
    }
    with _eda_lock:
        _eda_steps.setdefault(session_id, [])
        # Replace existing step or append
        steps = _eda_steps[session_id]
        for i, s in enumerate(steps):
            if s["id"] == step_id:
                steps[i] = event
                return
        steps.append(event)

# Import our real Kaggle tools module
from tools import (
    kaggle_auth_setup, 
    kaggle_search_tool, 
    kaggle_download_tool, 
    kaggle_push_tool,
    kaggle_status_tool,
    kaggle_output_tool,
    hf_search_tool,
    hf_download_tool,
    arxiv_search_tool
)

# ─────────────────────────────────────────────
# 0.  Environment Setup
#     Write ~/.kaggle/kaggle.json at startup so all subprocesses authenticate
# ─────────────────────────────────────────────
load_dotenv()
kaggle_auth_setup()   # ← Critical: must run before any kaggle call

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
GROQ_MODEL   = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


# ─────────────────────────────────────────────
# 1.  Shared State Definition
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    session_id: str
    run_id: str
    user_query:       str            # Raw user prompt
    architecture:     str            # Text architecture
    graph_architecture_json: dict    # NEW: React Flow JSON (nodes and edges)
    is_architecture_modified: bool   # NEW: Flag if user edited the graph manually
    
    # HPT fields
    hpt_search_space: Optional[dict]
    hpt_best_params: Optional[dict]
    hpt_best_value: Optional[float]
    hpt_trials: Optional[list]
    
    # Training fields
    epoch_metrics: Optional[list]
    architecture_desc: Optional[str]
    groq_commentary: Optional[str]
    
    # Graph fields
    final_graph: Optional[dict]
    
    # Metadata
    kaggle_results: str
    hf_results: str
    dataset_metadata: Optional[dict]
    
    dataset_options:  List[dict]     # [{title, ref, url, source, reason}, ...]
    dataset_selection_error: Optional[str]
    selected_dataset: str            # The Kaggle/HF ref chosen by the human
    dataset_csv_path:  str            # ← local path to downloaded CSV
    dataset_download_result: Optional[dict]
    dataset_validation_result: Optional[dict]
    dataset_acquisition_error: Optional[dict]
    execution_validation_result: Optional[dict]
    code_validation_result: Optional[dict]
    training_codegen_error: Optional[dict]
    
    eda_data: dict
    eda_narration: str
    eda_error: Optional[dict]
    training_config: Optional[dict]   # Human-defined training hyperparameters
    
    execution_mode:    str            # ← "local" or "cloud"
    generated_code:    Optional[str]  # Full PyTorch script produced by Engineer
    groq_fixed_code: Optional[str]
    active_code_source: Optional[str]
    code_revision: Optional[int]
    execution_success: Optional[bool]
    execution_output: Optional[str]
    
    training_logs:     str            # stdout + stderr (local or cloud)
    kernel_url:        str            # ← URL of the pushed Kaggle kernel
    kaggle_kernel_ref: str            # ← username/slug for polling
    retry_count:       int            # NEW: Loop limiter for debugger node
    metrics:           list           # NEW: List of epoch logs for real-time charting
    arxiv_benchmarks:  str            # NEW: Sprint 4 Literature comparator results
    deployment_artifacts: Optional[dict] # NEW: Export paths (onnx, torchscript, weights, api script)
    
    # Enterpise
    modality: str
    pipeline_config: dict
    imbalance: dict
    xai_report: Any
    
    # Tier 2 Run Fields
    problem_id: str
    input_data_version: int
    delta_state: Optional[dict]
    drift_report: Optional[dict]
    performance_decay_triggered: Optional[bool]
    comparison_report: Optional[str]
    run_manifest: Optional[dict]
    dataset_profile: Optional[dict]
    training_artifacts: Optional[dict]
    xai_artifacts: Optional[dict]
    fairness_artifacts: Optional[dict]
    benchmark_artifacts: Optional[dict]
    compliance_artifacts: Optional[list]
    plugin_artifacts: Optional[dict]


# ─────────────────────────────────────────────
# 1. Global Live Stream States
# ─────────────────────────────────────────────
hpt_state = {
  "current_trial": 0,
  "total_trials": 15,
  "best_value": 0.0,
  "best_params": {},
  "trials": [],
  "logs": [],
  "status": "running"
}

training_state = {
  "current_epoch": 0,
  "total_epochs": 50,
  "metrics": [],
  "architecture": "",
  "best_params": {},
  "groq_commentary": "",
  "logs": [],
  "status": "running"
}

MAX_CODEGEN_RETRIES = 3


# ─────────────────────────────────────────────
# 2. LLM Factory
# ─────────────────────────────────────────────
def get_llm(temperature: float = 0.5) -> ChatGroq:
    """Return a ChatGroq instance using openai/gpt-oss-120b."""
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=temperature,
        api_key=GROQ_API_KEY,
    )


# ─────────────────────────────────────────────
# 3.  Helper — resilient JSON parsing
# ─────────────────────────────────────────────
def _strip_code_fences(raw: str) -> str:
    """Remove ```python … ``` or ``` … ``` fences if present."""
    match = re.search(r"```(?:[pP]ython)?\n(.*?)```", raw, re.DOTALL)
    return match.group(1).strip() if match else raw.strip()

def _resilient_json_parse(raw: str) -> Optional[Any]:
    """
    Tries to parse JSON from a raw string that might contain LLM noise or minor syntax errors.
    1. Strips markdown fences.
    2. Extracts content between first { and last }.
    3. Fixes trailing commas.
    4. Falls back to returning None if parsing fails.
    """
    import json
    import re
    
    # Pre-cleaning
    raw = _strip_code_fences(raw).strip()
    
    # Find bounds
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end <= start:
        # Check for array if object not found
        start = raw.find("[")
        end = raw.rfind("]") + 1
        if start == -1 or end <= start:
            return None
            
    json_str = raw[start:end]
    
    # Try 1: Standard
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
        
    # Try 2: Fix trailing commas
    try:
        fixed = re.sub(r',\s*([\]}])', r'\1', json_str)
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
        
    # Try 3: Fix missing quotes around boolean/null or other very common LLM slips
    # but we'll stop here to avoid corrupting data.
    
    return None


# ─────────────────────────────────────────────
# 4.  Node 1 — Architect
# ─────────────────────────────────────────────
def architect_node(state: AgentState) -> dict:
    import json
    import chainlit as cl
    from groq import Groq
    
    problem = state.get("user_query", "binary classification")
    
    prompt = f"""You are a neural network architect.
Problem: {problem}

Return ONLY a JSON object. No markdown. No explanation. 
No code fences. Raw JSON only, starting with {{ and ending with }}.

{{
  "nodes": [
    {{"id":"1","type":"customNode","width":220,"height":80,
      "position":{{"x":300,"y":50}},
      "data":{{"label":"Input","nodeType":"Input",
               "params":{{"shape":"30,"}}}}}},
    {{"id":"2","type":"customNode","width":220,"height":80,
      "position":{{"x":300,"y":180}},
      "data":{{"label":"Dense_1","nodeType":"Dense",
               "params":{{"units":128,"activation":"relu"}}}}}},
    {{"id":"3","type":"customNode","width":220,"height":80,
      "position":{{"x":300,"y":310}},
      "data":{{"label":"BatchNorm_1","nodeType":"BatchNorm1d",
               "params":{{}}}}}},
    {{"id":"4","type":"customNode","width":220,"height":80,
      "position":{{"x":300,"y":440}},
      "data":{{"label":"Dropout_1","nodeType":"Dropout",
               "params":{{"rate":0.3}}}}}},
    {{"id":"5","type":"customNode","width":220,"height":80,
      "position":{{"x":300,"y":570}},
      "data":{{"label":"Dense_2","nodeType":"Dense",
               "params":{{"units":64,"activation":"relu"}}}}}},
    {{"id":"6","type":"customNode","width":220,"height":80,
      "position":{{"x":300,"y":700}},
      "data":{{"label":"Output","nodeType":"Output",
               "params":{{"units":1,"activation":"sigmoid"}}}}}}
  ],
  "edges": [
    {{"id":"e1-2","source":"1","target":"2","animated":true}},
    {{"id":"e2-3","source":"2","target":"3","animated":true}},
    {{"id":"e3-4","source":"3","target":"4","animated":true}},
    {{"id":"e4-5","source":"4","target":"5","animated":true}},
    {{"id":"e5-6","source":"5","target":"6","animated":true}}
  ],
  "rationale": "3-layer network for tabular binary classification."
}}

Adapt node count, units, and output activation to the problem.
Keep x=300 for all nodes. Increment y by 130 per node.
"""
    
    client = Groq(api_key=GROQ_API_KEY)
    graph = None
    attempts = 3
    
    for i in range(attempts):
        _safe_log(f"[architect] Synthesizing architecture (attempt {i+1}/{attempts})...")
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2 + (i * 0.2), # Increase temp on retry to get a different result
            max_tokens=2500,
        )
        raw = resp.choices[0].message.content.strip()
        graph = _resilient_json_parse(raw)
        
        if graph and "nodes" in graph:
            break
        _safe_log("[architect] ⚠️ JSON parse failed, retrying...")
    
    if not graph:
        _safe_log("[architect] 🛑 Failed to synthesize valid JSON architecture. Using safe fallback.")
        graph = {
            "nodes": [
                {"id":"1","type":"customNode","width":220,"height":80,"position":{"x":300,"y":50},"data":{"label":"Input","nodeType":"Input","params":{"shape":"30,"}}},
                {"id":"2","type":"customNode","width":220,"height":80,"position":{"x":300,"y":200},"data":{"label":"Output","nodeType":"Output","params":{"units":1,"activation":"sigmoid"}}}
            ],
            "edges": [{"id":"e1-2","source":"1","target":"2","animated":True}],
            "rationale": "Safe fallback architecture (Linear probe)."
        }
    
    # Store in BOTH places so nothing loses it
    cl.user_session.set("pending_graph", graph)
    cl.user_session.set("architect_graph", graph)
    
    return {
        "architecture_options": [graph],
        "graph_architecture_json": graph,
        "is_architecture_modified": False,
        "architecture": f"CustomGraph | {graph.get('rationale', '')}"
    }


# ─────────────────────────────────────────────
# 4b.  HITL React Flow Pause Node
# ─────────────────────────────────────────────
def hitl_model_pause_node(state: AgentState) -> dict:
    """
    UI-managed pause barrier.
    Chainlit resumes this node via update_state(..., as_node="hitl_model_pause").
    """
    return {}


# ─────────────────────────────────────────────
# 4c.  Dataset Sourcer Nodes (Parallel)
# ─────────────────────────────────────────────
def _extract_keyword(query: str, llm) -> str:
    kw_prompt = textwrap.dedent(f"""
        Given this user query: "{query}"
        Output exactly 3 short search keywords separated by spaces only (no commas, no punctuation).
        Example: wind turbine sensor
    """).strip()
    kw_response  = llm.invoke(kw_prompt)
    raw_kw       = kw_response.content.strip().split("\n")[0]
    clean_kw     = raw_kw.replace(",", " ").replace(";", " ").replace(".", " ")
    return " ".join(clean_kw.split()[:3])

def kaggle_sourcer_node(state: AgentState) -> dict:
    """Searches Kaggle for real datasets based on the user query."""
    llm = get_llm(temperature=0.0)
    search_query = _extract_keyword(state['user_query'], llm)
    _safe_log(f"[kaggle_sourcer] Searching for: '{search_query}'")
    
    res = kaggle_search_tool.invoke({"query": search_query})
    return {"kaggle_results": res}

def huggingface_sourcer_node(state: AgentState) -> dict:
    """Searches HuggingFace for real datasets based on the user query."""
    llm = get_llm(temperature=0.0)
    search_query = _extract_keyword(state['user_query'], llm)
    _safe_log(f"[hf_sourcer] Searching for: '{search_query}'")
    
    res = hf_search_tool.invoke({"query": search_query})
    return {"hf_results": res}


UNSUPPORTED_TABULAR_KEYWORDS = [
    "image",
    "images",
    "histopath",
    "histology",
    "microscopy",
    "xray",
    "x-ray",
    "mri",
    "ct scan",
    "dicom",
    "segmentation",
    "breakhis",
]


def _annotate_dataset_candidate(candidate: dict) -> dict:
    annotated = dict(candidate)
    haystack = " ".join(
        str(candidate.get(key, "")) for key in ("title", "ref", "description", "url")
    ).lower()
    supported = not any(keyword in haystack for keyword in UNSUPPORTED_TABULAR_KEYWORDS)
    annotated["expected_modality"] = "tabular"
    annotated["supported_by_current_pipeline"] = supported
    annotated["reason"] = (
        "Compatible with the current tabular CSV workflow."
        if supported
        else "Excluded because the current v1 workflow supports tabular CSV datasets only."
    )
    return annotated


def _build_dataset_acquisition_error(kind: str, message: str, ref: str = "") -> dict:
    return {
        "kind": kind,
        "dataset_ref": ref,
        "message": message,
    }


def _validation_failure(kind: str, message: str, ref: str = "", path: str = "") -> dict:
    return {
        "dataset_validation_result": {
            "status": "failed",
            "kind": kind,
            "message": message,
            "resolved_path": path,
        },
        "dataset_acquisition_error": _build_dataset_acquisition_error(kind, message, ref),
    }

def dataset_ranker_node(state: AgentState) -> dict:
    """Resolve candidate datasets and keep only tabular-compatible options."""
    import json
    kr = state.get("kaggle_results", "[]")
    
    try: kr_json = json.loads(kr)
    except: kr_json = []
    
    combined = [_annotate_dataset_candidate(c) for c in kr_json if "error" not in c]
    
    arch_full = "Custom Visual Graph" 
    
    if not combined:
        return {
            "architecture": arch_full,
            "dataset_selection_error": "Kaggle search authentication or network failed.",
            "dataset_options": [{
                "title": "⚠️ Kaggle Database Search Failed",
                "ref":   "error/error",
                "url":   "https://kaggle.com",
                "source": "kaggle",
                "reason": "Search authentication or network failed.",
                "supported_by_current_pipeline": False,
                "expected_modality": "tabular",
            }]
        }

    supported_candidates = [c for c in combined if c.get("supported_by_current_pipeline")]
    if not supported_candidates:
        return {
            "architecture": arch_full,
            "dataset_selection_error": "No tabular CSV-compatible datasets were found for the current workflow.",
            "dataset_options": [],
        }

    llm = get_llm(temperature=0.0)
    prompt = textwrap.dedent(f"""
        User Problem: "{state['user_query']}"
        Available Datasets:
        {json.dumps(supported_candidates, indent=2)}
        
        Rank the top 3 best and most relevant tabular CSV-compatible datasets to solve the User Problem.
        Output exactly one valid JSON array of strings containing the 'ref' of the chosen datasets in order of preference.
        Example: ["owner/dataset1", "dataset2", "another-org/dataset3"]
    """).strip()
    
    response = llm.invoke(prompt)
    top_refs = _resilient_json_parse(response.content)
    if not top_refs or not isinstance(top_refs, list):
        _safe_log("[dataset_ranker] ⚠️ JSON parse failed for rankings, using fallback first ref.")
        top_refs = [supported_candidates[0]["ref"]] if supported_candidates else []
        
    dataset_options = []
    for ref in top_refs[:3]:
        for c in supported_candidates:
            if c["ref"] == ref:
                chosen = dict(c)
                chosen["reason"] = f"Top selected dataset for '{state['user_query']}'"
                dataset_options.append(chosen)
                break
                
    if not dataset_options and supported_candidates:
        c = dict(supported_candidates[0])
        c["reason"] = "Fallback allocation"
        dataset_options = [c]

    return {
        "architecture": arch_full,
        "dataset_selection_error": None,
        "dataset_options": dataset_options
    }



# ─────────────────────────────────────────────
# 4b.  HITL Pause Node
# ─────────────────────────────────────────────
def hitl_pause_node(state: AgentState) -> dict:
    """
    UI-managed pause barrier.
    Chainlit resumes this node via update_state(..., as_node="hitl_pause").
    """
    return {}


# ─────────────────────────────────────────────
# 5.  Node 2 — Dataset Downloader (NEW — REAL)
# ─────────────────────────────────────────────
def _legacy_dataset_downloader_node_string(state: AgentState) -> dict:
    """
    Downloads the selected Kaggle or HF dataset to disk.
    Stores the absolute local CSV path in state so the Engineer can use it directly.
    """
    ref = state["selected_dataset"]
    _safe_log(f"[downloader] Downloading dataset: {ref}")
    
    # Identify source from options
    source = "kaggle"
    opts = state.get("dataset_options", [])
    for opt in opts:
        if opt["ref"] == ref:
            source = opt.get("source", "kaggle")
            break

    if source == "huggingface":
        csv_path = hf_download_tool.invoke({"dataset_ref": ref})
    else:
        csv_path = kaggle_download_tool.invoke({"dataset_ref": ref})

    if isinstance(csv_path, str) and csv_path[:5] == "ERROR":
        _safe_log(f"[downloader] Download failed: {csv_path}")
        csv_path = ""

    _safe_log(f"[downloader] ✅ CSV ready at: {csv_path}")
    dataset_profile = DatasetProfile(
        source="kaggle" if ref and "/" in ref else "huggingface",
        dataset_ref=ref or "",
        csv_path=csv_path,
        provenance={"summary": f"Downloaded dataset reference: {ref or 'local_upload'}"},
        lineage={"input_data_version": state.get("input_data_version", 1)},
    )
    return {"dataset_csv_path": csv_path, "dataset_profile": dataset_profile.model_dump(mode="json")}


# ─────────────────────────────────────────────
# 5b.  Node — EDA Analyzer (Sprint 5)
# ─────────────────────────────────────────────
def dataset_downloader_node(state: AgentState) -> dict:
    """
    Download the selected dataset and return a structured acquisition result.
    """
    ref = state["selected_dataset"]
    _safe_log(f"[downloader] Downloading dataset: {ref}")

    source = "kaggle"
    for opt in state.get("dataset_options", []):
        if opt.get("ref") == ref:
            source = opt.get("source", "kaggle")
            break

    if source == "huggingface":
        download_result = hf_download_tool.invoke({"dataset_ref": ref})
    else:
        download_result = kaggle_download_tool.invoke({"dataset_ref": ref})

    if not isinstance(download_result, dict):
        download_result = {
            "status": "download_failed",
            "source": source,
            "dataset_ref": ref,
            "resolved_path": "",
            "detected_format": "",
            "error_message": f"Downloader returned unexpected payload: {type(download_result).__name__}",
        }

    csv_path = download_result.get("resolved_path", "")
    status = download_result.get("status", "download_failed")
    if status == "ok":
        _safe_log(f"[downloader] CSV ready at: {csv_path}")
    else:
        _safe_log(f"[downloader] Download failed: {download_result.get('error_message', 'unknown error')}")

    dataset_profile = DatasetProfile(
        source=download_result.get("source", source),
        dataset_ref=ref or "",
        csv_path=csv_path,
        modality="tabular",
        provenance={
            "summary": f"Selected dataset reference: {ref or 'local_upload'}",
            "download_status": status,
            "download_error": download_result.get("error_message", ""),
        },
        lineage={"input_data_version": state.get("input_data_version", 1)},
    )

    state_update = {
        "dataset_csv_path": csv_path,
        "dataset_download_result": download_result,
        "dataset_profile": dataset_profile.model_dump(mode="json"),
    }
    if status != "ok":
        state_update["dataset_acquisition_error"] = _build_dataset_acquisition_error(
            status,
            download_result.get("error_message", "Dataset download failed."),
            ref,
        )
    return state_update


def dataset_validation_node(state: AgentState) -> dict:
    """
    Validate that the downloaded artifact exists and is compatible with the v1 tabular flow.
    """
    import pandas as pd
    from pathlib import Path

    ref = state.get("selected_dataset", "")
    download_result = state.get("dataset_download_result") or {}
    status = download_result.get("status", "download_failed")
    if status != "ok":
        return _validation_failure(
            status,
            download_result.get("error_message", "Dataset download failed."),
            ref,
            download_result.get("resolved_path", ""),
        )

    resolved_path = download_result.get("resolved_path", "")
    if not resolved_path:
        return _validation_failure("download_failed", "Dataset download did not produce a usable file path.", ref)

    path = Path(resolved_path)
    if not path.exists():
        return _validation_failure("download_failed", f"Downloaded file was not found on disk: {resolved_path}", ref, resolved_path)

    suffix = path.suffix.lower()
    if suffix not in {".csv", ".tsv"}:
        return _validation_failure(
            "unsupported_format",
            f"Downloaded artifact is '{suffix or 'unknown'}', but the current workflow supports tabular CSV-compatible datasets only.",
            ref,
            resolved_path,
        )

    try:
        sample = pd.read_csv(path, sep=None, engine="python", nrows=100)
    except Exception as exc:
        return _validation_failure(
            "unsupported_format",
            f"Dataset could not be parsed as tabular CSV input: {exc}",
            ref,
            resolved_path,
        )

    if sample.empty or len(sample.columns) == 0:
        return _validation_failure(
            "unsupported_format",
            "Dataset parsed successfully but contained no usable tabular columns.",
            ref,
            resolved_path,
        )

    dataset_profile = DatasetProfile.model_validate(state.get("dataset_profile") or {})
    dataset_profile.csv_path = resolved_path
    dataset_profile.modality = "tabular"
    dataset_profile.col_count = len(sample.columns)
    dataset_profile.columns = [str(col) for col in sample.columns]

    return {
        "dataset_csv_path": resolved_path,
        "dataset_profile": dataset_profile.model_dump(mode="json"),
        "dataset_validation_result": {
            "status": "ok",
            "kind": "tabular_csv",
            "resolved_path": resolved_path,
            "detected_modality": "tabular",
            "detected_format": suffix.lstrip("."),
            "column_count": len(sample.columns),
            "sample_row_count": len(sample),
        },
        "dataset_acquisition_error": None,
    }


def check_dataset_validation_status(state: AgentState) -> str:
    validation = state.get("dataset_validation_result") or {}
    if validation.get("status") == "ok":
        return "continue"
    return "retry_dataset_selection"


def eda_analyzer_node(state: AgentState) -> dict:
    """
    Profile the dataset (Stats, Bins, Corr) and generate AI narration.
    Emits real-time progress events into _eda_steps for SSE streaming.
    """
    import pandas as pd
    import numpy as np
    import json

    # Resolve session_id from state (best effort)
    session_id = state.get("session_id", "default")

    csv_path = state.get("dataset_csv_path")
    if not csv_path or not os.path.exists(csv_path):
        _eda_emit(session_id, "load", "Loading Dataset", status="error",
                  detail="CSV file not found on disk.", pct=0)
        with _eda_lock:
            _eda_done[session_id] = True
        return {
            "eda_data": {},
            "eda_narration": "Data source not found.",
            "eda_error": {"path": csv_path or "", "message": "CSV file not found on disk."},
        }

    # ── Clear previous run ──────────────────────────────────────────────────
    with _eda_lock:
        _eda_steps[session_id] = []
        _eda_done[session_id]  = False

    _safe_log(f"[eda] Profiling dataset: {csv_path}")

    # ── Step 1: Load CSV ────────────────────────────────────────────────────
    _eda_emit(session_id, "load", "Loading CSV", status="running",
              detail=f"Reading {os.path.basename(csv_path)} …", pct=5)
    try:
        df = pd.read_csv(csv_path, sep=None, engine='python')
        n_rows, n_cols = df.shape
        _eda_emit(session_id, "load", "Loading CSV", status="done",
                  detail=f"{n_rows:,} rows × {n_cols} columns loaded", pct=12,
                  data={"rows": n_rows, "cols": n_cols})
    except Exception as e:
        _safe_log(f"[eda] CSV read error: {e}")
        _eda_emit(session_id, "load", "Loading CSV", status="error",
                  detail=str(e), pct=0)
        with _eda_lock:
            _eda_done[session_id] = True
        return {
            "eda_data": {},
            "eda_narration": f"Error reading data: {e}",
            "eda_error": {"path": csv_path, "message": str(e)},
        }

    # ── Step 2: Sample for Speed ────────────────────────────────────────────
    _eda_emit(session_id, "sample", "Sampling Data", status="running",
              detail="Capping at 20,000 rows for speed …", pct=15)
    if len(df) > 20000:
        df_sample = df.sample(20000, random_state=42)
        _eda_emit(session_id, "sample", "Sampling Data", status="done",
                  detail=f"Sampled 20,000 / {n_rows:,} rows", pct=20)
    else:
        df_sample = df
        _eda_emit(session_id, "sample", "Sampling Data", status="done",
                  detail=f"Using all {n_rows:,} rows", pct=20)

    # ── Step 3: Missing Value Analysis ─────────────────────────────────────
    _eda_emit(session_id, "missing", "Analyzing Missing Values", status="running",
              detail="Scanning for nulls, NaNs, blanks …", pct=22)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(1)
    missing_report = {
        col: {"count": int(missing[col]), "pct": float(missing_pct[col])}
        for col in df.columns if missing[col] > 0
    }
    _eda_emit(session_id, "missing", "Analyzing Missing Values", status="done",
              detail=f"{len(missing_report)} columns with missing data", pct=30,
              data={"missing": missing_report})

    # ── Step 4: Numerical Distributions ────────────────────────────────────
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    _eda_emit(session_id, "dist", "Computing Distributions", status="running",
              detail=f"Histograms for {len(numeric_cols)} numeric columns …", pct=32)
    distributions = {}
    for i, col in enumerate(numeric_cols[:15]):
        clean_col = df_sample[col].dropna()
        if not clean_col.empty:
            counts, bins = np.histogram(clean_col, bins=20)
            distributions[col] = {
                "counts": counts.tolist(),
                "bins": bins.tolist(),
                "mean":  round(float(clean_col.mean()), 4),
                "std":   round(float(clean_col.std()), 4),
                "min":   round(float(clean_col.min()), 4),
                "max":   round(float(clean_col.max()), 4),
                "skew":  round(float(clean_col.skew()), 3),
            }
        pct_done = 32 + int((i + 1) / max(len(numeric_cols[:15]), 1) * 20)
        _eda_emit(session_id, "dist", "Computing Distributions", status="running",
                  detail=f"  ↳ {col} ({i+1}/{len(numeric_cols[:15])})", pct=pct_done)
    _eda_emit(session_id, "dist", "Computing Distributions", status="done",
              detail=f"{len(distributions)} histograms computed", pct=52)

    # ── Step 5: Categorical Distributions ──────────────────────────────────
    categorical_cols = df_sample.select_dtypes(include=['object', 'category']).columns.tolist()
    _eda_emit(session_id, "cat", "Encoding Categoricals", status="running",
              detail=f"Value counts for {len(categorical_cols)} categorical columns …", pct=54)
    categoricals = {}
    for col in categorical_cols[:10]:
        counts = df_sample[col].value_counts().head(10)
        safe_labels = [str(lbl)[:50] + ("…" if len(str(lbl)) > 50 else "") for lbl in counts.index.tolist()]
        categoricals[col] = {"labels": safe_labels, "values": counts.values.tolist()}
    _eda_emit(session_id, "cat", "Encoding Categoricals", status="done",
              detail=f"{len(categoricals)} categorical profiles ready", pct=62)

    # ── Step 6: Correlation Matrix ──────────────────────────────────────────
    _eda_emit(session_id, "corr", "Computing Correlations", status="running",
              detail=f"Pearson correlation across {len(numeric_cols)} features …", pct=64)
    corr_matrix = df_sample[numeric_cols].corr().round(2).fillna(0).to_dict()

    # Find top correlations
    top_corrs = []
    keys = list(corr_matrix.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            v = corr_matrix[keys[i]].get(keys[j], 0)
            if abs(v) > 0.5 and keys[i] != keys[j]:
                top_corrs.append({"a": keys[i], "b": keys[j], "r": v})
    top_corrs.sort(key=lambda x: abs(x["r"]), reverse=True)

    _eda_emit(session_id, "corr", "Computing Correlations", status="done",
              detail=f"{len(top_corrs)} strong correlations (|r|>0.5) found", pct=75,
              data={"top_corrs": top_corrs[:5]})

    # ── Step 7: Outlier Detection ───────────────────────────────────────────
    _eda_emit(session_id, "outlier", "Detecting Outliers", status="running",
              detail="IQR fence method on numeric columns …", pct=77)
    outlier_report = {}
    for col in numeric_cols[:10]:
        col_data = df_sample[col].dropna()
        if col_data.empty:
            continue
        q1, q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        iqr = q3 - q1
        n_out = int(((col_data < (q1 - 1.5 * iqr)) | (col_data > (q3 + 1.5 * iqr))).sum())
        if n_out > 0:
            outlier_report[col] = {"count": n_out, "pct": round(n_out / len(col_data) * 100, 1)}
    _eda_emit(session_id, "outlier", "Detecting Outliers", status="done",
              detail=f"{len(outlier_report)} features with outliers detected", pct=84,
              data={"outliers": outlier_report})

    # ── Step 8: Groq Narration ──────────────────────────────────────────────
    _eda_emit(session_id, "narrate", "Generating AI Insights", status="running",
              detail="Groq LLM analyzing patterns and generating insights …", pct=86)
    llm = get_llm(temperature=0.0)
    summary_stats = df_sample.describe().to_string()
    narration_prompt = f"""
        You are an AI Data Scientist. Analyze this dataset for the task: {state['user_query']}

        Summary Stats:
        {summary_stats[:2000]}

        Missing Values: {len(missing_report)} columns affected.
        Top Correlations: {top_corrs[:3]}
        Outlier Columns: {list(outlier_report.keys())[:5]}

        Provide 4-5 sharp, numbered insights about:
        1. Data quality issues
        2. Key feature distributions
        3. Important correlations
        4. Recommended preprocessing steps
        5. Model architecture implications

        Be concise, direct, and technically precise.
    """
    narration = llm.invoke(narration_prompt).content
    _eda_emit(session_id, "narrate", "Generating AI Insights", status="done",
              detail="Analysis complete — insights ready", pct=98)

    # ── Done ────────────────────────────────────────────────────────────────
    with _eda_lock:
        _eda_done[session_id] = True

    eda_payload = {
            "columns":       df.columns.tolist(),
            "row_count":     len(df),
            "col_count":     len(df.columns),
            "distributions": distributions,
            "categoricals":  categoricals,
            "correlation":   corr_matrix,
            "missing":       missing_report,
            "outliers":      outlier_report,
            "top_corrs":     top_corrs[:10],
        }
    bundle = load_or_create_bundle(state)
    dataset_profile = DatasetProfile.model_validate(state.get("dataset_profile") or {})
    dataset_profile.row_count = len(df)
    dataset_profile.col_count = len(df.columns)
    dataset_profile.columns = df.columns.tolist()
    dataset_profile.missing = missing_report
    dataset_profile.top_correlations = top_corrs[:10]
    dataset_profile.detected_sensitive_features = [col for col in df.columns if any(token in col.lower() for token in ("gender", "sex", "race", "ethnicity", "age"))]
    profile_path = os.path.join(bundle.run_manifest.paths["artifacts"], "dataset_profile.json")
    write_json(profile_path, dataset_profile.model_dump(mode="json"))
    register_artifact(bundle.run_manifest, "dataset_profile", "dataset_profile", profile_path)

    return {
        "eda_data": eda_payload,
        "eda_narration": narration,
        "dataset_profile": dataset_profile.model_dump(mode="json"),
        "run_manifest": bundle.run_manifest.model_dump(mode="json"),
    }


# ─────────────────────────────────────────────
# 5c.  HITL — EDA Dashboard Pause
# ─────────────────────────────────────────────
def hitl_eda_pause_node(state: AgentState) -> dict:
    """
    UI-managed pause barrier.
    Chainlit resumes this node via update_state(..., as_node="hitl_eda_pause").
    """
    return {}


# ─────────────────────────────────────────────
# 5d.  HITL — Execution Choice Node (NEW)
# ─────────────────────────────────────────────
def execution_choice_node(state: AgentState) -> dict:
    """
    UI-managed pause barrier.
    Chainlit resumes this node via update_state(..., as_node="execution_choice").
    """
    return {}


# ─────────────────────────────────────────────
# 6a. HPT Node
# ─────────────────────────────────────────────
def hpt_node(state: AgentState) -> dict:
    """Derive HPT search space from graph architecture."""
    final_graph = state.get("graph_architecture_json", {})
    nodes = final_graph.get("nodes", [])
    
    space = {
        "learning_rate": ("float_log", 1e-5, 1e-2),
        "batch_size":    ("categorical", [16, 32, 64, 128]),
        "optimizer":     ("categorical", ["adam", "sgd", "rmsprop"])
    }
    
    parts = []
    for node in sorted(nodes, key=lambda n: n.get("position", {}).get("y", 0)):
        t = node.get("data", {}).get("nodeType", "Dense")
        nid = node.get("id", "")
        p = node.get("data", {}).get("params", {})
        
        if t == "Dense":
            space[f"units_{nid}"] = ("categorical", [32,64,128,256,512])
            space[f"activation_{nid}"] = ("categorical", ["relu","tanh","selu"])
            parts.append(f"Dense({p.get('units','?')},{p.get('activation','relu')})")
        elif t == "Dropout":
            space[f"rate_{nid}"] = ("float", 0.1, 0.6)
            parts.append(f"Dropout({p.get('rate','?')})")
        elif t == "LSTM":
            space[f"lstm_units_{nid}"] = ("categorical", [32,64,128,256])
            space[f"lstm_return_seq_{nid}"] = ("categorical", [True, False])
            parts.append(f"LSTM({p.get('units','?')})")
        elif t == "Conv1D":
            space[f"filters_{nid}"] = ("categorical", [16,32,64,128])
            space[f"kernel_{nid}"] = ("categorical", [3, 5, 7])
            parts.append(f"Conv1D")
        elif t == "BatchNorm1d":
            space[f"bn_momentum_{nid}"] = ("float", 0.01, 0.5)
            parts.append("BN1d")
        elif t == "Output":
            parts.append(f"Output({p.get('units','?')},{p.get('activation','sigmoid')})")

    return {
        "hpt_search_space": space,
        "final_graph": final_graph,
        "architecture_desc": " → ".join(parts)
    }

def _validate_and_fix_syntax(code: str, llm, max_attempts: int = 3) -> str:
    """AST-parse script; if SyntaxError detected, ask Groq to fix ONLY the syntax."""
    import ast
    for attempt in range(max_attempts):
        try:
            ast.parse(code)
            return code          # valid — done
        except SyntaxError as exc:
            _safe_log(f"[syntax-fix] Attempt {attempt+1}/{max_attempts} — SyntaxError at line {exc.lineno}: {exc.msg}")
            context_start = max(0, (exc.lineno or 1) - 5)
            context_lines = code.splitlines()[context_start : (exc.lineno or 1) + 3]
            context_snippet = "\n".join(f"{context_start + i + 1}: {l}" for i, l in enumerate(context_lines))
            fix_prompt = textwrap.dedent(f"""
                You are a Python syntax expert. Fix ONLY the syntax error below.
                DO NOT change any logic, imports, or variable names.
                Return ONLY the corrected complete Python script. No markdown fences.

                SyntaxError: {exc.msg} at line {exc.lineno}
                Context (lines {context_start+1}-{(exc.lineno or 1)+3}):
                {context_snippet}

                FULL SCRIPT TO FIX:
                {code}
            """).strip()
            try:
                response = llm.invoke(fix_prompt)
                code = _strip_code_fences(response.content)
            except Exception as e:
                _safe_log(f"[syntax-fix] LLM call failed: {e}")
                break
    return code  # return best effort


def _validate_code_contract(script: str) -> dict:
    import ast

    if not script or not script.strip():
        return {"status": "failed", "message": "Generated training code is empty."}

    try:
        ast.parse(script)
    except SyntaxError as exc:
        return {
            "status": "failed",
            "message": f"Generated training code is not valid Python: {exc.msg} at line {exc.lineno}",
        }

    required_markers = ('"type":"epoch_metric"', '"type": "epoch_metric"', "epoch_metric")
    if not any(marker in script for marker in required_markers):
        return {
            "status": "failed",
            "message": "Generated training code does not emit required epoch metrics.",
        }

    return {"status": "ok", "message": "Generated training code parsed successfully and preserves required telemetry."}


# ─────────────────────────────────────────────
# 6b. Engineer Agent — DETERMINISTIC TEMPLATE
# ─────────────────────────────────────────────
def _build_model_class(graph_nodes):
    """Build a PyTorch nn.Module class from the visual graph nodes."""
    sorted_nodes = sorted(graph_nodes, key=lambda n: n.get("position", {}).get("y", 0))

    init_lines = []
    forward_lines = []
    layer_idx = 0
    prev_dim = "input_dim"

    for node in sorted_nodes:
        d = node.get("data", {})
        ntype = d.get("nodeType", "")
        params = d.get("params", {})

        if ntype == "Input":
            continue  # input_dim handled externally
        elif ntype == "Dense":
            units = params.get("units", 128)
            act   = params.get("activation", "relu")
            init_lines.append(f"        self.fc{layer_idx} = nn.Linear({prev_dim}, {units})")
            if act == "relu":
                init_lines.append(f"        self.act{layer_idx} = nn.ReLU()")
            elif act == "tanh":
                init_lines.append(f"        self.act{layer_idx} = nn.Tanh()")
            elif act == "selu":
                init_lines.append(f"        self.act{layer_idx} = nn.SELU()")
            else:
                init_lines.append(f"        self.act{layer_idx} = nn.ReLU()")
            forward_lines.append(f"        x = self.fc{layer_idx}(x)")
            forward_lines.append(f"        x = self.act{layer_idx}(x)")
            prev_dim = str(units)
            layer_idx += 1
        elif ntype == "BatchNorm1d":
            init_lines.append(f"        self.bn{layer_idx} = nn.BatchNorm1d({prev_dim})")
            forward_lines.append(f"        x = self.bn{layer_idx}(x)")
            layer_idx += 1
        elif ntype == "Dropout":
            rate = params.get("rate", 0.3)
            init_lines.append(f"        self.drop{layer_idx} = nn.Dropout({rate})")
            forward_lines.append(f"        x = self.drop{layer_idx}(x)")
            layer_idx += 1
        elif ntype == "Output":
            # Always use num_classes for CrossEntropyLoss compatibility
            init_lines.append(f"        self.output_layer = nn.Linear({prev_dim}, num_classes)")
            forward_lines.append(f"        x = self.output_layer(x)")
            prev_dim = "num_classes"

    if not init_lines:
        # Fallback: simple 2-layer net
        init_lines = [
            "        self.fc0 = nn.Linear(input_dim, 128)",
            "        self.act0 = nn.ReLU()",
            "        self.drop0 = nn.Dropout(0.3)",
            "        self.fc1 = nn.Linear(128, 64)",
            "        self.act1 = nn.ReLU()",
            "        self.output_layer = nn.Linear(64, num_classes)",
        ]
        forward_lines = [
            "        x = self.fc0(x)",
            "        x = self.act0(x)",
            "        x = self.drop0(x)",
            "        x = self.fc1(x)",
            "        x = self.act1(x)",
            "        x = self.output_layer(x)",
        ]

    return "\n".join(init_lines), "\n".join(forward_lines)


def engineer_node(state: AgentState) -> dict:
    from anomallm.engineer import engineer_node as _egn
    return _egn(state)


# ─────────────────────────────────────────────
# 6c. Groq Loopfixer — now a pass-through
#     (deterministic template doesn't need fixing)
# ─────────────────────────────────────────────
def groq_loopfixer_node(state: AgentState) -> dict:
    script = state.get("generated_code", "")
    if not script:
        return {}
    validation = _validate_code_contract(script)
    revision = int(state.get("code_revision") or 0) + 1
    return {
        "generated_code": script,
        "groq_fixed_code": None,
        "active_code_source": "loopfixer",
        "code_revision": revision,
        "code_validation_result": validation,
        "training_codegen_error": None,
    }

# ─────────────────────────────────────────────
# 7.  Execution Sandbox
# ─────────────────────────────────────────────
async def execution_sandbox_node(state: AgentState) -> dict:
    import json
    import asyncio
    import os
    import tempfile
    global hpt_state, training_state
    
    # Dynamically update total epochs display for frontend
    tc = state.get("training_config") or {}
    training_state["total_epochs"] = int(tc.get("epochs", 50))
    
    import ast
    script = state.get("generated_code", "") or state.get("groq_fixed_code", "")
    code_source = state.get("active_code_source") or ("generated_code" if state.get("generated_code") else "groq_fixed_code")
    code_revision = int(state.get("code_revision") or 0)
    if not script or not script.strip():
        return {
            **state,
            "training_logs": "ERROR: Engineer agent did not produce code.",
            "metrics": [],
            "execution_validation_result": {
                "status": "failed",
                "message": "Engineer agent did not produce executable code.",
            },
            "code_validation_result": {"status": "failed", "message": "Engineer agent did not produce executable code."},
        }

    _safe_log(f"[sandbox] Executing code source={code_source} revision={code_revision}")

    # — PRE-FLIGHT: AST syntax check before writing to disk —
    llm_for_fix = get_llm(temperature=0.0)
    preflight_source = code_source
    preflight_revision = code_revision
    try:
        ast.parse(script)
    except SyntaxError as se:
        _safe_log(f"[sandbox] SyntaxError detected pre-flight at line {se.lineno}: {se.msg} — auto-repairing...")
        script = _validate_and_fix_syntax(script, llm_for_fix)
        try:
            ast.parse(script)
            _safe_log("[sandbox] Pre-flight syntax repair succeeded.")
            preflight_source = "sandbox_preflight"
            preflight_revision = code_revision + 1
        except SyntaxError as se2:
            err = f"FATAL SyntaxError after repair: {se2.msg} at line {se2.lineno}"
            _safe_log(f"[sandbox] {err}")
            return {
                **state,
                "training_logs": err,
                "metrics": [],
                "execution_validation_result": {
                    "status": "failed",
                    "message": err,
                },
                "code_validation_result": {"status": "failed", "message": err},
                "training_codegen_error": {"status": "failed", "message": err},
            }



    _safe_log("[sandbox] 💻 Local mode selected. Subprocess streaming enabled...")
    stdout_lines = []
    
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(script)
            tmp_path = tmp.name
            
        with open("last_run_script.py", "w", encoding="utf-8") as f:
            f.write(script)

        process = await asyncio.create_subprocess_exec(
            sys.executable, tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=os.getcwd()
        )

        async def fetch_commentary(mets):
            if not mets: return
            llm = get_llm(temperature=0.4)
            p = f"You are a senior ML training monitor. Given these epoch metrics for a {state.get('architecture_desc','')} model, write ONE sentence of actionable commentary about what you observe. Be specific about the numbers. Focus on: overfitting signs, learning rate issues, convergence speed.\\nMetrics: {json.dumps(mets)}"
            c = _strip_code_fences(llm.invoke(p).content)
            training_state["groq_commentary"] = c

        import json as _json
        metrics_list = []
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            
            raw = line.decode('utf-8', errors='ignore').strip()
            if not raw:
                continue
            
            stdout_lines.append(raw)
            if len(stdout_lines) > 5000:
                stdout_lines = stdout_lines[-5000:]
            
            training_state["logs"].append(raw)
            if len(training_state["logs"]) > 50:
                training_state["logs"] = training_state["logs"][-50:]
            
            try:
                parsed = _json.loads(raw)
                t = parsed.get("type", "")
                if t == "epoch_metric":
                    metric = {
                        "epoch":    int(parsed.get("epoch", 0)),
                        "loss":     float(parsed.get("loss", 0)),
                        "val_loss": float(parsed.get("val_loss", 0)),
                        "acc":      float(parsed.get("acc", 0)),
                        "val_acc":  float(parsed.get("val_acc", 0)),
                    }
                    metrics_list.append(metric)
                    training_state["metrics"] = metrics_list
                    training_state["current_epoch"] = metric["epoch"]
                    
                    if len(metrics_list) % 5 == 0:
                        asyncio.create_task(fetch_commentary(metrics_list[-5:]))
                elif t == "hpt_trial":
                    hpt_state["current_trial"] = int(parsed.get("trial", 0))
                    hpt_state["total_trials"] = parsed.get("total", 15)
                    hpt_state["best_value"] = float(parsed.get("best_so_far", 0.0))
                    hpt_state["current_params"] = parsed.get("params", {})
                    hpt_state["trials"].append(parsed)
                    hpt_state["logs"].append(f"> Trial {parsed.get('trial')} complete: value={parsed.get('value', 0):.4f}")
                elif t == "hpt_complete":
                    hpt_state["status"] = "complete"
                    hpt_state["best_params"] = parsed.get("best_params", {})
                    training_state["best_params"] = parsed.get("best_params", {})
            except (_json.JSONDecodeError, ValueError, TypeError):
                pass
        
        await process.wait()
        logs = "\\n".join(stdout_lines[-5000:])
    except Exception as exc:
        logs = f"ERROR: Subprocess failed — {exc}"
    finally:
        try: os.remove(tmp_path)
        except: pass

    bundle = load_or_create_bundle(state)
    paths = ensure_run_paths(bundle.run_manifest.run_id)
    logs_path = os.path.join(paths.logs, "training.log")
    write_text(logs_path, logs)

    metrics_list = metrics_list if 'metrics_list' in locals() else []
    artifact_metrics = metrics_list[-1] if metrics_list else {}
    training_artifacts = TrainingArtifacts.model_validate(state.get("training_artifacts") or {})
    training_artifacts.training_logs_path = logs_path
    training_artifacts.metrics = artifact_metrics
    training_artifacts.training_config = state.get("training_config") or {}
    training_artifacts.predictions_path = os.path.join(paths.artifacts, "predictions.csv")
    training_artifacts.evaluation_path = os.path.join(paths.artifacts, "evaluation.json")
    training_artifacts.export_paths = {
        "model": os.path.join(paths.exports, "model.pt"),
        "scripted": os.path.join(paths.exports, "model_scripted.pt"),
        "onnx": os.path.join(paths.exports, "model.onnx"),
        "meta": os.path.join(paths.exports, "model_meta.json"),
    }
    training_artifacts.plots = {
        "loss_curve": os.path.join(paths.plots, "loss_curve.png"),
        "feature_correlation": os.path.join(paths.plots, "telemetry_distribution.png"),
        "shap_importance": os.path.join(paths.plots, "shap_importance.png"),
    }
    if os.path.exists(training_artifacts.evaluation_path):
        try:
            with open(training_artifacts.evaluation_path, "r", encoding="utf-8") as handle:
                evaluation_payload = json.load(handle)
            training_artifacts.model_card = {
                "target_column": evaluation_payload.get("target_column"),
                "feature_names": evaluation_payload.get("feature_columns", []),
                "top_features": evaluation_payload.get("feature_importance", []),
            }
            training_artifacts.best_params = state.get("hpt_best_params") or {}
            training_artifacts.task_type = evaluation_payload.get("metrics", {}).get("task_type", "unknown")
        except Exception:
            pass
    register_artifact(bundle.run_manifest, "training_log", "log", logs_path)
    return {
        "generated_code": script,
        "groq_fixed_code": None,
        "active_code_source": preflight_source,
        "code_revision": preflight_revision,
        "code_validation_result": _validate_code_contract(script),
        "training_logs": logs,
        "metrics": metrics_list,
        "execution_validation_result": {
            "status": "ok" if metrics_list else "failed",
            "message": (
                "Training execution produced epoch metrics."
                if metrics_list
                else "Training execution completed without emitting required epoch metrics."
            ),
        },
        "run_manifest": bundle.run_manifest.model_dump(mode="json"),
        "training_artifacts": training_artifacts.model_dump(mode="json"),
    }


# ─────────────────────────────────────────────
# 8b.  Node — Debugger (Auto-Healing)
# ─────────────────────────────────────────────
def debugger_node(state: AgentState) -> dict:
    """
    Regenerate the deterministic training template and validate it.
    """
    from anomallm.engineer import engineer_node as deterministic_engineer_node

    regenerated = deterministic_engineer_node(state)
    new_code = regenerated.get("generated_code", "")
    validation = _validate_code_contract(new_code)
    revision = int(state.get("code_revision") or 0) + 1
    failure_message = validation.get("message", "Deterministic debugger regeneration produced invalid code.")

    return {
        "generated_code": new_code,
        "groq_fixed_code": None,
        "active_code_source": "debugger",
        "code_revision": revision,
        "retry_count": state.get("retry_count", 0) + 1,
        "training_logs": "",
        "code_validation_result": validation,
        "execution_validation_result": None,
        "training_codegen_error": (
            None if validation.get("status") == "ok"
            else {"status": "failed", "message": failure_message}
        ),
    }


def codegen_contract_validator_node(state: AgentState) -> dict:
    """Ensure generated code still emits required training telemetry before execution."""
    script = state.get("generated_code", "") or state.get("groq_fixed_code", "")
    validation = _validate_code_contract(script)
    failure_message = validation.get("message", "Generated training code failed validation.")
    return {
        "code_validation_result": validation,
        "training_codegen_error": (
            None if validation.get("status") == "ok"
            else {"status": "failed", "message": failure_message}
        ),
        "training_logs": (
            state.get("training_logs", "")
            if validation.get("status") == "ok"
            else f"ERROR: {failure_message}"
        ),
    }


def validate_execution_output_node(state: AgentState) -> dict:
    """Validate training outputs before downstream analytics/reporting."""
    metrics = state.get("metrics") or []
    required_metric_keys = {"epoch", "loss", "val_loss", "acc", "val_acc"}

    if not metrics:
        return {
            "execution_validation_result": {
                "status": "failed",
                "message": "Training execution completed without emitting required epoch metrics.",
            },
            "training_logs": (
                (state.get("training_logs", "") + "\n") if state.get("training_logs") else ""
            ) + "ERROR: Training execution completed without emitting required epoch metrics.",
        }

    last_metric = metrics[-1]
    missing_keys = sorted(required_metric_keys - set(last_metric.keys()))
    if missing_keys:
        return {
            "execution_validation_result": {
                "status": "failed",
                "message": (
                    "Training metrics are missing required keys: " + ", ".join(missing_keys)
                ),
            },
            "training_logs": (
                (state.get("training_logs", "") + "\n") if state.get("training_logs") else ""
            ) + "ERROR: Training metrics are missing required keys.",
        }

    run_id = state.get("run_id", "")
    required_artifacts = [
        os.path.join(os.getcwd(), "runs", run_id, "artifacts", "metrics.json"),
        os.path.join(os.getcwd(), "runs", run_id, "artifacts", "evaluation.json"),
    ]
    missing_artifacts = [os.path.basename(path) for path in required_artifacts if not os.path.exists(path)]
    if missing_artifacts:
        return {
            "execution_validation_result": {
                "status": "failed",
                "message": (
                    "Training execution completed but required artifacts are missing: "
                    + ", ".join(missing_artifacts)
                ),
            },
            "training_logs": (
                (state.get("training_logs", "") + "\n") if state.get("training_logs") else ""
            ) + "ERROR: Training execution completed but required artifacts are missing.",
        }

    return {
        "execution_validation_result": {
            "status": "ok",
            "message": "Training execution produced usable metrics and artifacts.",
        }
    }


# ─────────────────────────────────────────────
# 8d.  Node — ArXiv Comparator (Sprint 4)
# ─────────────────────────────────────────────
def arxiv_comparator_node(state: AgentState) -> dict:
    """
    Search ArXiv for scholarly literature, analyze benchmarks, and perform gap analysis.
    """
    global training_state
    training_state["logs"].append("📡 Searching ArXiv for scholarly benchmarks...")
    llm = get_llm(temperature=0.0)
    
    # ── Step A: Refine scholarly search query ────────────────────────────────
    query_prompt = f"""
        Given the following user ML objective, generate a precise scholarly search string for the ArXiv API.
        Objective: {state['user_query']}
        Focus on: Methodology, Deep Learning, Benchmarks.
        Output ONLY the search string (e.g., "heart failure prediction neural network").
    """
    search_query = llm.invoke(query_prompt).content.strip().replace('"', '')
    _safe_log(f"[comparator] 📡 ArXiv Search Query: {search_query}")
    
    # ── Step B: Invoke tool ──────────────────────────────────────────────────
    literature_raw = arxiv_search_tool.invoke(search_query)
    if literature_raw.startswith("ERROR"):
        return {"arxiv_benchmarks": "No relevant scholarly literature found on ArXiv."}
        
    # ── Step C: Scientific Analysis (Groq) ───────────────────────────────────
    import json
    metrics_str = "Not Available"
    if state.get('metrics'):
        metrics_str = json.dumps(state['metrics'][-1:])

    analysis_prompt = f"""
        You are a Senior AI Research Scientist. Analyze the following ArXiv abstracts relevant to: {state['user_query']}
        Current Model Architecture: {state.get('architecture', 'Custom Visual Graph')}
        Current Dataset: {state.get('selected_dataset', 'Self-curated')}
        Final Model Metrics (OmniML): {metrics_str}

        === SCHOLARLY ABSTRACTS ===
        {literature_raw}

        Perform a LITERATURE COMPARISON & GAP ANALYSIS:
        1. Summarize the methodologies in these papers.
        2. Extract reported metrics (Accuracy, MSE, AUC, etc.) AS-IS from the abstracts. 
           ★ NEGATIVE CONSTRAINT: If a paper does not provide explicit numeric results, state 'No quantitative benchmarks available for this paper'. DO NOT estimate or hallucinate numbers from the text.
        3. Compare the OmniML model's Final Model Metrics from above directly against the quantitative ArXiv results.
        4. For every paper/metric, add a one-line "Comparability Note" explaining if it can be directly compared to our model's performance.
        5. Produce a 'Gap Analysis': What SOTA features or data techniques mentioned in these papers are missing from our current implementation?

        Format the output in professional Markdown.
    """
    
    comp_response = llm.invoke(analysis_prompt)
    arxiv_markdown = comp_response.content
    
    return {"arxiv_benchmarks": arxiv_markdown}


# ─────────────────────────────────────────────
# 8d½. Model Deployer — Export + API Generation
# ─────────────────────────────────────────────
def model_deployer_node(state: AgentState) -> dict:
    """
    Generates production deployment artifacts from the trained model exports:
    - FastAPI inference script (using ONNX Runtime)
    - Dockerfile
    - requirements.txt
    """
    import os, json

    export_dir = os.path.join(os.getcwd(), "exports")
    meta_path  = os.path.join(export_dir, "model_meta.json")

    # Read model metadata
    if not os.path.exists(meta_path):
        _safe_log("[deployer] ⚠️  No model_meta.json found — skipping deployment generation")
        return {"deployment_artifacts": None}

    with open(meta_path) as f:
        meta = json.load(f)

    input_dim     = meta.get("input_dim", 1)
    num_classes   = meta.get("num_classes", 2)
    feature_names = meta.get("feature_names", [f"f{i}" for i in range(input_dim)])
    target_col    = meta.get("target_col", "target")
    val_acc       = meta.get("final_val_acc", 0.0)

    # — 1. Generate FastAPI inference script —
    example_payload = ", ".join([f'"{fn}": 0.0' for fn in feature_names[:6]])
    if len(feature_names) > 6:
        example_payload += ", ..."

    api_script = f'''"""
OmniML Inference API — Auto-Generated
======================================
Serves predictions via ONNX Runtime for maximum speed.
Model: {target_col} classifier ({num_classes} classes, {input_dim} features)
Validation accuracy: {val_acc:.4f}

Start:  uvicorn serve_api:app --host 0.0.0.0 --port 8080
Test:   curl -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d '{{"features": [0.0, ...]}}'
"""
import os
import json
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

app = FastAPI(
    title="OmniML Inference API",
    description="Auto-generated model serving endpoint",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ONNX model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.onnx")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

FEATURE_NAMES = {json.dumps(feature_names)}
NUM_CLASSES = {num_classes}
INPUT_DIM = {input_dim}


class PredictRequest(BaseModel):
    features: List[float]

    class Config:
        json_schema_extra = {{
            "example": {{"features": [0.0] * min(input_dim, 10)}}
        }}


class PredictResponse(BaseModel):
    predicted_class: int
    confidence: float
    probabilities: List[float]


@app.get("/health")
def health():
    return {{"status": "ok", "model": "onnx", "input_dim": INPUT_DIM, "num_classes": NUM_CLASSES}}


@app.get("/meta")
def model_meta():
    return {{
        "feature_names": FEATURE_NAMES,
        "input_dim": INPUT_DIM,
        "num_classes": NUM_CLASSES,
    }}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.features) != INPUT_DIM:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {{INPUT_DIM}} features, got {{len(req.features)}}"
        )

    x = np.array([req.features], dtype=np.float32)
    logits = session.run(None, {{input_name: x}})[0][0]

    # Softmax
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()

    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    return PredictResponse(
        predicted_class=pred_class,
        confidence=round(confidence, 6),
        probabilities=[round(float(p), 6) for p in probs],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
'''

    api_path = os.path.join(export_dir, "serve_api.py")
    with open(api_path, "w") as f:
        f.write(api_script)
    _safe_log(f"[deployer] ✅ FastAPI script → {api_path}")

    # — 2. Generate requirements.txt —
    reqs = """fastapi>=0.104.0
uvicorn>=0.24.0
onnxruntime>=1.16.0
numpy>=1.24.0
pydantic>=2.0.0
"""
    reqs_path = os.path.join(export_dir, "requirements.txt")
    with open(reqs_path, "w") as f:
        f.write(reqs)

    # — 3. Generate Dockerfile —
    dockerfile = """FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.onnx .
COPY model_meta.json .
COPY serve_api.py .

EXPOSE 8080
CMD ["uvicorn", "serve_api:app", "--host", "0.0.0.0", "--port", "8080"]
"""
    df_path = os.path.join(export_dir, "Dockerfile")
    with open(df_path, "w") as f:
        f.write(dockerfile)

    _safe_log(f"[deployer] ✅ Dockerfile → {df_path}")
    _safe_log(f"[deployer] ✅ requirements.txt → {reqs_path}")

    artifacts = {
        "export_dir":    export_dir,
        "weights_path":  os.path.join(export_dir, "model.pt"),
        "onnx_path":     os.path.join(export_dir, "model.onnx"),
        "torchscript_path": os.path.join(export_dir, "model_scripted.pt"),
        "api_path":      api_path,
        "dockerfile":    df_path,
        "requirements":  reqs_path,
        "meta":          meta,
    }
    return {"deployment_artifacts": artifacts}


# ─────────────────────────────────────────────
# 8e.  Node 5 — Evaluator Agent
# ─────────────────────────────────────────────
def evaluator_node(state: AgentState) -> dict:
    """
    Reads the training_logs and produces a structured professional Markdown report and PDF.
    """
    import os
    import json
    from fpdf import FPDF
    import textwrap

    llm = get_llm(temperature=0.5)

    eval_prompt = textwrap.dedent(f"""
        You are the Evaluator Agent for OmniML, an enterprise-grade Auto-ML system.

        ── Run Context ──────────────────────────────────────────────────────────────
        User Query       : {state['user_query']}
        Architecture     : {state['architecture']}
        Dataset Ref      : {state['selected_dataset']}
        Local CSV Path   : {state.get('dataset_csv_path', 'N/A')}
        Training Logs (includes Optuna tuning):
        {state['training_logs']}

        ── Scholarly Comparison (Sprint 4) ──────────────────────────────────────────
        {state.get('arxiv_benchmarks', 'Literature search skipped.')}
        ─────────────────────────────────────────────────────────────────────────────

        ── Explainable AI (XAI) Narrative ────────────────────────────────────────
        {state.get('xai_report', 'XAI analysis skipped.')}
        ─────────────────────────────────────────────────────────────────────────────

        Generate a VAST, comprehensive, and highly detailed professional Markdown report using EXACTLY this structure:
        
        # 🤖 OmniML — Autonomous ML Run Report

        ## 1. 🧠 Architecture Selection
        ...
        
        ## 3. 📊 Smart Training & Tuning
        ...

        ## 4. 📚 Literature Benchmarking & Gap Analysis
        Include a verbatim block of the scholarly comparison provided in the context.
        Ensure you focus on the 'Gap Analysis' to show where the model stands.

        ## 5. ✅ Verdict & Next Steps
        Provide a sweeping, clear deployment recommendation. List 3-4 highly detailed, concrete next steps to improve the model's robustness or explainability.

        ---
        *Report generated by OmniML | Powered by Groq openai/gpt-oss-120b*
    """).strip()

    report = llm.invoke(eval_prompt)
    report_content = report.content
    
    # ── Generate PDF ────────────────────────────────────────────────────────
    try:
        class UnicodePDF(FPDF):
            def __init__(self):
                super().__init__()
                try:
                    self.add_font("DejaVu", "", 
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
                        uni=True)
                    self.add_font("DejaVu", "B",
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                        uni=True)
                    self.unicode_font = "DejaVu"
                except Exception:
                    self.unicode_font = None

            def safe_cell(self, w, h, txt, **kwargs):
                if self.unicode_font:
                    self.set_font(self.unicode_font, size=11)
                    self.cell(w, h, txt, **kwargs)
                else:
                    safe = txt.encode("latin-1", "ignore").decode("latin-1")
                    self.cell(w, h, safe, **kwargs)

            def safe_multi_cell(self, w, h, txt, **kwargs):
                if self.unicode_font:
                    self.set_font(self.unicode_font, size=11)
                    self.multi_cell(w, h, txt, **kwargs)
                else:
                    safe = txt.encode("latin-1", "ignore").decode("latin-1")
                    self.multi_cell(w, h, safe, **kwargs)

            def header(self):
                self.set_font('helvetica', 'B', 15)
                self.safe_cell(0, 10, 'OmniML Technical Report', border=False, ln=1, align='C')
                self.ln(5)

            def footer(self):
                self.set_y(-15)
                self.set_font('helvetica', 'I', 8)
                self.safe_cell(0, 10, f'Page {self.page_no()}', border=False, ln=1, align='C')

        pdf = UnicodePDF()
        pdf.add_page()
        pdf.set_font('helvetica', size=11)
        
        # Write abstract/metrics if JSON exists
        metrics_text = ""
        if os.path.exists("metrics.json"):
            with open("metrics.json", "r") as f:
                data = json.load(f)
                metrics_text = json.dumps(data, indent=4)
            pdf.set_font('helvetica', 'B', 12)
            pdf.safe_cell(0, 10, "Optimized Model Metrics & Hyperparameters:", ln=1)
            pdf.set_font('Courier', size=10)
            pdf.safe_multi_cell(0, 7, metrics_text)
            pdf.ln(5)
            
        pdf.set_font('helvetica', size=11)
        
        # Process simple markdown to text for PDF compatibility
        for line in report_content.split("\n"):
            line = line.strip()
            if not line:
                pdf.ln(2)
                continue
                
            if line.startswith("# "):
                pdf.set_font('helvetica', 'B', 16)
                pdf.safe_cell(0, 12, line[2:], ln=1)
            elif line.startswith("## "):
                pdf.set_font('helvetica', 'B', 14)
                pdf.safe_cell(0, 10, line[3:], ln=1)
            elif line.startswith("### "):
                pdf.set_font('helvetica', 'B', 12)
                pdf.safe_cell(0, 8, line[4:], ln=1)
            else:
                pdf.set_font('helvetica', size=11)
                pdf.safe_multi_cell(0, 6, line)
            pdf.ln(1)
                
        # Append graphical analytics page
        pdf.add_page()
        pdf.set_font('helvetica', 'B', 16)
        pdf.safe_cell(0, 10, "Smart Training & Analytics", ln=1, align='C')
        pdf.ln(10)
        
        y_pos = pdf.get_y()
        if os.path.exists("telemetry_distribution.png"):
            pdf.set_font('helvetica', 'B', 12)
            pdf.safe_cell(0, 10, "1. Multi-Feature Correlation Matrix", ln=1)
            pdf.image("telemetry_distribution.png", w=160)
            y_pos = pdf.get_y() + 10
            
        if os.path.exists("loss_curve.png"):
            if y_pos > 180:
                pdf.add_page()
                y_pos = pdf.get_y()
            else:
                pdf.set_y(y_pos)
            pdf.set_font('helvetica', 'B', 12)
            pdf.safe_cell(0, 10, "2. Hyperparameter Optimization History", ln=1)
            pdf.image("loss_curve.png", w=160)
            y_pos = pdf.get_y() + 10

        if os.path.exists("shap_importance.png"):
            if y_pos > 180:
                pdf.add_page()
                y_pos = pdf.get_y()
            else:
                pdf.set_y(y_pos)
            pdf.set_font('helvetica', 'B', 12)
            pdf.safe_cell(0, 10, "4. SHAP Feature Importance", ln=1)
            pdf.image("shap_importance.png", w=160)
            y_pos = pdf.get_y() + 10

        if os.path.exists("confusion_matrix.png"):
            if y_pos > 180:
                pdf.add_page()
                y_pos = pdf.get_y()
            else:
                pdf.set_y(y_pos)
            pdf.set_font('helvetica', 'B', 12)
            pdf.safe_cell(0, 10, "3. Classification Confusion Matrix", ln=1)
            pdf.image("confusion_matrix.png", w=140)

        pdf.output("Final_Report.pdf")
        _safe_log("[evaluator] ✅ PDF report generated successfully.")
    except Exception as e:
        _safe_log(f"[evaluator] ❌ PDF generation failed: {e}")

    return {"final_report": report_content}

def check_execution_success(state: AgentState) -> str:
    """
    Checks if local execution produced a real crash, routing to debugger if so. 
    Respects a maximum of 3 retries.
    Only triggers on actual Python tracebacks, not incidental 'Error' in log text.
    """
    logs = state.get("training_logs", "")
    retries = state.get("retry_count", 0)
    mode = state.get("execution_mode", "local")
    
    if mode == "local" and retries < 3:
        # Only trigger debugger for actual Python tracebacks
        if "Traceback (most recent call last)" in logs:
            return "debugger"
        # Also catch SyntaxError at top level
        if logs.strip().startswith("SyntaxError") or "\nSyntaxError" in logs:
            return "debugger"
        if not (state.get("metrics") or []):
            return "debugger"
            
    return "execution_output_validator"


def check_execution_validation_status(state: AgentState) -> str:
    validation = state.get("execution_validation_result") or {}
    if validation.get("status") == "ok":
        return "continue"
    if state.get("retry_count", 0) >= MAX_CODEGEN_RETRIES:
        return "execution_failure"
    return "debugger"


def check_codegen_validation_status(state: AgentState) -> str:
    validation = state.get("code_validation_result") or {}
    if validation.get("status") == "ok":
        return "continue"
    if state.get("retry_count", 0) >= MAX_CODEGEN_RETRIES:
        return "execution_failure"
    return "debugger"

def check_engineer_code(state: AgentState) -> str:
    generated_code = state.get("generated_code")
    if not generated_code or not generated_code.strip():
        return "hitl_model_pause"
    return "execution_choice"

# ─────────────────────────────────────────────
# Tier 2 — Autonomous Ecology Nodes
# ─────────────────────────────────────────────
def run_history_node(state: AgentState) -> dict:
    from anomallm.persistence import SQLiteDataLayer
    
    # 1. Generate a UNIQUE problem_id per run using UUID
    #    Also generate a semantic label via Groq for display purposes
    llm = get_llm(temperature=0.0)
    query = state.get("user_query", "")
    
    # Generate a stable semantic label for display
    canonicalizer_prompt = textwrap.dedent(f"""
        You are the OmniML Canonicalizer. Your goal is to map a user query to a stable, 2-4 word snake_case ML Problem Label.
        Ignore all conversational filler like "I need an AI for", "Please build", or "Help me with".
        FOCUS purely on the core ML task and target domain.
        
        Examples:
        - "I want to predict titanic survival" -> titanic_survival
        - "Classify insurance fraud in biometric records" -> insurance_fraud_biometrics
        - "Help me build a breast cancer biopsy diagnostic" -> breast_cancer_biopsy
        - "Build a house price predictor" -> house_price_prediction
        
        User Query: {query}
        Return ONLY the snake_case label.
    """).strip()
    
    try:
        res = llm.invoke(canonicalizer_prompt)
        semantic_label = res.content.strip().lower()
        semantic_label = re.sub(r'[^a-z0-9_]', '', semantic_label.replace(' ', '_'))
    except Exception:
        pure_words = re.findall(r'[a-zA-Z0-9]+', query.lower())
        semantic_label = "_".join(pure_words[:5]) or "default_problem"
    
    # 2. Create a UNIQUE problem_id by appending a short UUID
    #    This ensures each new conversation/run gets its own identity
    short_uid = uuid.uuid4().hex[:8]
    problem_id = f"{semantic_label}_{short_uid}"
    
    _safe_log(f"[ml_ops] 🏰 Unique Problem ID: {problem_id} (semantic: {semantic_label})")
    
    # 3. No history lookup by semantic label anymore — each run is a fresh pioneer run
    #    This prevents false "Welcome Back" messages from unrelated previous sessions
    db = SQLiteDataLayer()
    run_id = problem_id
    manifest = create_run_manifest(run_id, query, problem_id)
    plugin_registry = PluginRegistry()
    enabled_plugins = ((state.get("training_config") or {}).get("plugin_config") or {}).get("enabled_plugins") or []
    plugin_artifacts = plugin_registry.execute_slot("pre_run", state, enabled_plugins=enabled_plugins).model_dump(mode="json")

    return {
        "run_id": run_id,
        "problem_id": problem_id,
        "input_data_version": 1,
        "delta_state": {},
        "run_manifest": manifest.model_dump(mode="json"),
        "plugin_artifacts": plugin_artifacts,
    }

def drift_sentry_node(state: AgentState) -> dict:
    from anomallm.drift import check_feature_drift
    
    version = state.get("input_data_version", 1)
    if version <= 1:
        # First upload: naturally no drift
        return {"drift_report": {"status": "no_drift", "features": {}}}
        
    delta = state.get("delta_state", {})
    ref_csv = delta.get("previous_csv_path")
    cur_csv = state.get("dataset_csv_path")
    
    _safe_log(f"[sentry] Performing statistical drift checks vs Reference Data (Version {version-1})...")
    
    if not ref_csv or not cur_csv:
        return {"drift_report": {"status": "error", "reason": "Missing raw CSV payload for comparison"}}
        
    report = check_feature_drift(ref_csv, cur_csv)
    return {"drift_report": report}

def hitl_drift_approval_node(state: AgentState) -> dict:
    """UI-managed pause barrier for drift approval."""
    return {}

def compare_runs_node(state: AgentState) -> dict:
    from anomallm.persistence import SQLiteDataLayer
    from anomallm.comparison import perform_comparative_rag
    
    db = SQLiteDataLayer()
    histories = db.get_run_histories_by_problem(state.get("problem_id", "default_problem"))
    metrics = state.get("metrics") or []

    if not metrics:
        return {
            "comparison_report": (
                "Comparison skipped because the current run did not produce usable training metrics."
            )
        }
    
    if len(histories) > 0:
        llm = get_llm(temperature=0.0)
        curr_metrics = metrics[-1]
        curr_report = state.get("final_report", "")
        comparison = perform_comparative_rag(llm, curr_metrics, curr_report, histories)
        return {"comparison_report": comparison}
    
    return {"comparison_report": "This is the pioneer run for this problem; no historical baselines found."}


def execution_failure_node(state: AgentState) -> dict:
    error = state.get("training_codegen_error") or {}
    message = error.get("message") or "Training code generation failed after the maximum retry budget."
    retries = state.get("retry_count", 0)
    source = state.get("active_code_source", "unknown")
    revision = state.get("code_revision", 0)
    return {
        "training_codegen_error": {
            "status": "failed",
            "message": message,
        },
        "final_report": (
            "## Training Code Generation Error\n\n"
            f"{message}\n\n"
            f"- Retry count: `{retries}`\n"
            f"- Active code source: `{source}`\n"
            f"- Code revision: `{revision}`\n\n"
            "The training pipeline stopped after exhausting the automatic repair budget."
        ),
    }

def check_drift_condition(state: AgentState) -> str:
    report = state.get("drift_report", {})
    if report.get("status") == "drift_detected":
        return "hitl_drift_approval"
    return "modality"

def save_run_history_node(state: AgentState) -> dict:
    from anomallm.persistence import SQLiteDataLayer
    
    _safe_log("[ml_ops] 💾 Saving run history to Tier 2 ML persistence...")
    db = SQLiteDataLayer()
    
    mets = state.get("metrics", [])
    val_acc = "0.0"
    if mets:
        val_acc = str(mets[-1].get("val_acc", "0.0"))
        
    db.create_run_history(
        problem_id=state.get("problem_id", "default_problem"),
        dataset_ref=state.get("selected_dataset", ""),
        data_version=str(state.get("input_data_version", 1)),
        dataset_csv_path=state.get("dataset_csv_path", ""),
        metrics=state.get("metrics", [])[-1] if state.get("metrics") else {},
        val_accuracy=val_acc,
        report=state.get("final_report", ""),
        run_manifest=state.get("run_manifest") or {},
        artifact_index={
            "dataset_profile": state.get("dataset_profile"),
            "training_artifacts": state.get("training_artifacts"),
            "xai_artifacts": state.get("xai_artifacts"),
            "fairness_artifacts": state.get("fairness_artifacts"),
            "benchmark_artifacts": state.get("benchmark_artifacts"),
            "compliance_artifacts": state.get("compliance_artifacts"),
        },
        template_types=[report.get("template_id") for report in (state.get("compliance_artifacts") or []) if isinstance(report, dict)],
    )
    return {}


def evaluator_node(state: AgentState) -> dict:
    """Assemble the primary markdown report from deterministic evidence artifacts."""
    bundle = load_or_create_bundle(state)
    bundle.dataset_profile = DatasetProfile.model_validate(state.get("dataset_profile") or {})
    bundle.training_artifacts = TrainingArtifacts.model_validate(state.get("training_artifacts") or {})
    report_lines = [
        "# OmniML Autonomous ML Run Report",
        "",
        "## Architecture Selection",
        "",
        f"- User query: {state.get('user_query', '')}",
        f"- Architecture: {state.get('architecture_desc') or state.get('architecture', 'N/A')}",
        f"- Dataset: {state.get('selected_dataset', 'N/A')}",
        "",
        "## Training Summary",
        "",
        f"- Metrics: {json.dumps(bundle.training_artifacts.metrics, indent=2) if bundle.training_artifacts.metrics else 'No metrics recorded.'}",
        f"- Best params: {json.dumps(bundle.training_artifacts.best_params, indent=2) if bundle.training_artifacts.best_params else 'No HPT params recorded.'}",
        "",
        "## Explainability",
        "",
        state.get("xai_report", "XAI analysis skipped."),
        "",
        "## Benchmarking",
        "",
        state.get("arxiv_benchmarks", "Benchmark analysis skipped."),
        "",
        "## Verdict",
        "",
        "Evidence-backed run summary generated from run-scoped artifacts. Review compliance attachments for regulated deployment contexts.",
    ]
    final_report = "\n".join(report_lines)
    report_path = os.path.join(bundle.run_manifest.paths["reports"], "final_report.md")
    write_text(report_path, final_report)
    register_artifact(bundle.run_manifest, "final_report", "report_markdown", report_path)
    return {"final_report": final_report, "run_manifest": bundle.run_manifest.model_dump(mode="json")}


def fairness_auditor_node(state: AgentState) -> dict:
    bundle = load_or_create_bundle(state)
    paths = ensure_run_paths(bundle.run_manifest.run_id)
    config = (state.get("training_config") or {}).get("fairness_config") or {}
    result = run_fairness_audit(
        state.get("dataset_csv_path", ""),
        os.path.join(paths.artifacts, "predictions.csv"),
        paths.artifacts,
        config=config,
    )
    register_artifact(bundle.run_manifest, "fairness_summary", "fairness", result.summary_path or os.path.join(paths.artifacts, "fairness_summary.json"))
    return {
        "fairness_artifacts": result.model_dump(mode="json"),
        "run_manifest": bundle.run_manifest.model_dump(mode="json"),
    }


def evidence_collector_node(state: AgentState) -> dict:
    bundle = load_or_create_bundle(state)
    bundle.dataset_profile = DatasetProfile.model_validate(state.get("dataset_profile") or {})
    bundle.training_artifacts = TrainingArtifacts.model_validate(state.get("training_artifacts") or {})
    if state.get("xai_artifacts"):
        bundle.xai_artifacts = XAIArtifacts.model_validate(state.get("xai_artifacts"))
    if state.get("fairness_artifacts"):
        bundle.fairness_artifacts = FairnessAuditResult.model_validate(state.get("fairness_artifacts"))
    if state.get("benchmark_artifacts"):
        bundle.benchmark_artifacts = BenchmarkArtifacts.model_validate(state.get("benchmark_artifacts"))
    if state.get("plugin_artifacts"):
        bundle.plugin_artifacts = PluginArtifacts.model_validate(state.get("plugin_artifacts"))
    registry = PluginRegistry()
    enabled_plugins = ((state.get("training_config") or {}).get("plugin_config") or {}).get("enabled_plugins") or []
    bundle.plugin_artifacts = registry.execute_slot("evidence_provider", state, bundle.plugin_artifacts, enabled_plugins=enabled_plugins)
    persist_bundle(bundle)
    return {
        "run_manifest": bundle.run_manifest.model_dump(mode="json"),
        "plugin_artifacts": bundle.plugin_artifacts.model_dump(mode="json"),
    }


def compliance_mapper_node(state: AgentState) -> dict:
    training_config = state.get("training_config") or {}
    compliance_modes = training_config.get("compliance_modes") or ["eu_ai_act", "fda_samd", "soc2"]
    return {"compliance_modes": compliance_modes}


def compliance_narrative_node(state: AgentState) -> dict:
    fairness_status = (state.get("fairness_artifacts") or {}).get("status", "not_available")
    narrative = (
        f"Compliance narratives are evidence-backed overlays. Fairness status for this run: {fairness_status}. "
        "Any missing evidence is disclosed explicitly in the rendered templates."
    )
    return {"compliance_narrative": narrative}


def compliance_renderer_node(state: AgentState) -> dict:
    bundle = load_or_create_bundle(state)
    bundle.dataset_profile = DatasetProfile.model_validate(state.get("dataset_profile") or {})
    bundle.training_artifacts = TrainingArtifacts.model_validate(state.get("training_artifacts") or {})
    if state.get("xai_artifacts"):
        bundle.xai_artifacts = XAIArtifacts.model_validate(state.get("xai_artifacts"))
    if state.get("fairness_artifacts"):
        bundle.fairness_artifacts = FairnessAuditResult.model_validate(state.get("fairness_artifacts"))
    if state.get("benchmark_artifacts"):
        bundle.benchmark_artifacts = BenchmarkArtifacts.model_validate(state.get("benchmark_artifacts"))
    report_templates = state.get("compliance_modes") or ["eu_ai_act", "fda_samd", "soc2"]
    reports = generate_compliance_reports(bundle, report_templates)
    bundle.compliance_artifacts = reports
    persist_bundle(bundle)
    return {
        "compliance_artifacts": [report.model_dump(mode="json") for report in reports],
        "run_manifest": bundle.run_manifest.model_dump(mode="json"),
    }


def arxiv_comparator_node(state: AgentState) -> dict:
    """Build benchmark evidence from Papers With Code plus ArXiv context."""
    global training_state
    training_state["logs"].append("Benchmark comparison running...")
    query = normalize_task_label(state.get("user_query", ""), state.get("selected_dataset", ""), state.get("modality", "tabular"))
    literature_raw = arxiv_search_tool.invoke(query.task_label)
    benchmark_config = (state.get("training_config") or {}).get("benchmark_config") or {}
    manager = BenchmarkSourceManager(
        timeout_seconds=int(benchmark_config.get("source_timeout_seconds", 10)),
        cache_ttl_days=int(benchmark_config.get("cache_ttl_days", 7)),
    )
    benchmark_artifacts = manager.resolve(
        query,
        literature_raw,
        mode=benchmark_config.get("mode", "prefer_live_then_cache"),
    )
    papers = benchmark_artifacts.cited_papers
    current_metrics = state.get("training_artifacts", {}).get("metrics") or (state.get("metrics", [{}])[-1] if state.get("metrics") else {})
    comparability = assess_comparability(query, current_metrics, benchmark_artifacts.leaderboard_entries, state.get("selected_dataset", ""))
    recommendations = build_gap_recommendations(comparability, papers, query.task_label)
    benchmark_artifacts.query = query
    benchmark_artifacts.comparability = comparability
    benchmark_artifacts.recommendations = recommendations
    benchmark_artifacts.rendered_markdown = render_benchmark_markdown(benchmark_artifacts)
    bundle = load_or_create_bundle(state)
    benchmark_path = os.path.join(bundle.run_manifest.paths["artifacts"], "benchmark_summary.json")
    write_json(benchmark_path, benchmark_artifacts.model_dump(mode="json"))
    register_artifact(bundle.run_manifest, "benchmark_summary", "benchmark", benchmark_path)
    return {
        "arxiv_benchmarks": benchmark_artifacts.rendered_markdown,
        "benchmark_artifacts": benchmark_artifacts.model_dump(mode="json"),
        "run_manifest": bundle.run_manifest.model_dump(mode="json"),
    }

# ─────────────────────────────────────────────
# 9.  Graph Assembly
# ─────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    # These nodes are UI-managed pause barriers. The app resumes them via
    # update_state(..., as_node="<pause_node>"), so they must not call interrupt().
    UI_MANAGED_INTERRUPTS = [
        "hitl_model_pause",
        "hitl_pause",
        "hitl_eda_pause",
        "hitl_drift_approval",
        "execution_choice",
    ]
    
    # Core Nodes
    graph.add_node("architect",          architect_node)
    graph.add_node("hitl_model_pause",   hitl_model_pause_node)
    graph.add_node("kaggle_sourcer",     kaggle_sourcer_node)
    graph.add_node("dataset_ranker",     dataset_ranker_node)
    graph.add_node("hitl_pause",         hitl_pause_node)
    from anomallm.nodes import modality_node, imbalance_node, xai_node
    graph.add_node("dataset_downloader", dataset_downloader_node)
    graph.add_node("dataset_validation", dataset_validation_node)
    graph.add_node("modality",           modality_node)
    graph.add_node("imbalance",          imbalance_node)
    graph.add_node("xai_node",           xai_node)
    graph.add_node("eda_analyzer",       eda_analyzer_node)
    graph.add_node("hitl_eda_pause",     hitl_eda_pause_node)
    graph.add_node("execution_choice",   execution_choice_node)
    graph.add_node("hpt",                hpt_node)
    graph.add_node("engineer",           engineer_node)
    graph.add_node("groq_loopfixer",     groq_loopfixer_node)
    graph.add_node("codegen_contract_validator", codegen_contract_validator_node)
    graph.add_node("execution_sandbox",  execution_sandbox_node)
    graph.add_node("execution_output_validator", validate_execution_output_node)
    graph.add_node("debugger",           debugger_node)
    graph.add_node("execution_failure",  execution_failure_node)
    graph.add_node("arxiv_comparator",   arxiv_comparator_node)
    graph.add_node("model_deployer",     model_deployer_node)
    graph.add_node("evaluator",          evaluator_node)
    graph.add_node("fairness_auditor",   fairness_auditor_node)
    graph.add_node("evidence_collector", evidence_collector_node)
    graph.add_node("compliance_mapper",  compliance_mapper_node)
    graph.add_node("compliance_narrative", compliance_narrative_node)
    graph.add_node("compliance_renderer", compliance_renderer_node)
    
    # Tier 2 Nodes
    graph.add_node("run_history",        run_history_node)
    graph.add_node("drift_sentry",       drift_sentry_node)
    graph.add_node("hitl_drift_approval", hitl_drift_approval_node)
    graph.add_node("save_run_history",   save_run_history_node)
    graph.add_node("compare_runs",       compare_runs_node)

    # ── Flow ──
    graph.set_entry_point("run_history")
    graph.add_edge("run_history",        "architect")
    graph.add_edge("architect",          "hitl_model_pause")
    graph.add_edge("hitl_model_pause",   "kaggle_sourcer")
    graph.add_edge("kaggle_sourcer",     "dataset_ranker")
    graph.add_edge("dataset_ranker",     "hitl_pause")
    graph.add_edge("hitl_pause",         "dataset_downloader")
    
    # Drift Check intercept
    graph.add_edge("dataset_downloader", "dataset_validation")
    graph.add_conditional_edges(
        "dataset_validation",
        check_dataset_validation_status,
        {"retry_dataset_selection": "hitl_pause", "continue": "drift_sentry"}
    )
    graph.add_conditional_edges("drift_sentry", check_drift_condition, {"hitl_drift_approval": "hitl_drift_approval", "modality": "modality"})
    graph.add_edge("hitl_drift_approval", "modality")
    
    graph.add_edge("modality",           "eda_analyzer")
    graph.add_edge("eda_analyzer",       "imbalance")
    graph.add_edge("imbalance",          "hitl_eda_pause")
    graph.add_edge("hitl_eda_pause",     "execution_choice")
    graph.add_edge("execution_choice",   "hpt")
    graph.add_edge("hpt",                "engineer")
    
    graph.add_conditional_edges(
        "engineer",
        check_engineer_code,
        {"execution_choice": "groq_loopfixer", "hitl_model_pause": "hitl_model_pause"}
    )
    
    graph.add_edge("groq_loopfixer",     "codegen_contract_validator")
    graph.add_conditional_edges(
        "codegen_contract_validator",
        check_codegen_validation_status,
        {"debugger": "debugger", "continue": "execution_sandbox", "execution_failure": "execution_failure"}
    )
    
    graph.add_conditional_edges(
        "execution_sandbox",
        check_execution_success,
        {"debugger": "debugger", "execution_output_validator": "execution_output_validator"}
    )
    graph.add_conditional_edges(
        "execution_output_validator",
        check_execution_validation_status,
        {"debugger": "debugger", "continue": "xai_node", "execution_failure": "execution_failure"}
    )
    graph.add_edge("debugger",           "codegen_contract_validator")
    graph.add_edge("xai_node",           "arxiv_comparator")
    graph.add_edge("arxiv_comparator",   "model_deployer")
    graph.add_edge("model_deployer",     "evaluator")
    
    # Tier 2 Closing Flow
    graph.add_edge("evaluator",          "fairness_auditor")
    graph.add_edge("fairness_auditor",   "evidence_collector")
    graph.add_edge("evidence_collector", "compliance_mapper")
    graph.add_edge("compliance_mapper",  "compliance_narrative")
    graph.add_edge("compliance_narrative", "compliance_renderer")
    graph.add_edge("compliance_renderer", "save_run_history")
    graph.add_edge("save_run_history",   "compare_runs")
    graph.add_edge("compare_runs",       END)
    graph.add_edge("execution_failure",  END)

    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()

    return graph.compile(
        checkpointer=memory,
        interrupt_before=UI_MANAGED_INTERRUPTS,
    )
