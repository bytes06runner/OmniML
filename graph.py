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
import subprocess
import textwrap
import tempfile
import threading
from typing import TypedDict, List, Optional

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt

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

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL   = "openai/gpt-oss-120b"


# ─────────────────────────────────────────────
# 1.  Shared State Definition
# ─────────────────────────────────────────────
class AgentState(TypedDict):
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
    selected_dataset: str            # The Kaggle/HF ref chosen by the human
    dataset_csv_path:  str            # ← local path to downloaded CSV
    
    eda_data: dict
    eda_narration: str
    training_config: Optional[dict]   # Human-defined training hyperparameters
    
    execution_mode:    str            # ← "local" or "cloud"
    generated_code:    Optional[str]  # Full PyTorch script produced by Engineer
    groq_fixed_code: Optional[str]
    execution_success: Optional[bool]
    execution_output: Optional[str]
    
    training_logs:     str            # stdout + stderr (local or cloud)
    kernel_url:        str            # ← URL of the pushed Kaggle kernel
    kaggle_kernel_ref: str            # ← username/slug for polling
    retry_count:       int            # NEW: Loop limiter for debugger node
    metrics:           list           # NEW: List of epoch logs for real-time charting
    arxiv_benchmarks:  str            # NEW: Sprint 4 Literature comparator results


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
# 3.  Helper — strip markdown fences from LLM code output
# ─────────────────────────────────────────────
def _strip_code_fences(raw: str) -> str:
    """Remove ```python … ``` or ``` … ``` fences if present."""
    match = re.search(r"```(?:[pP]ython)?\n(.*?)```", raw, re.DOTALL)
    return match.group(1).strip() if match else raw.strip()


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
    
    try:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2500,
        )
        raw = resp.choices[0].message.content.strip()
        
        # Aggressively strip any markdown fences Groq adds
        raw = raw.strip()
        if "```" in raw:
            parts = raw.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    raw = part
                    break
        
        # Find first { and last } to extract pure JSON
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start != -1 and end > start:
            raw = raw[start:end]
        
        graph = json.loads(raw)
    except Exception as e:
        # Hard fallback — always give a valid graph
        print(f"[ARCHITECT FALLBACK] {str(e)}", flush=True)
        graph = {
          "nodes": [
            {"id":"1","type":"customNode","width":220,"height":80,
             "position":{"x":300,"y":50},
             "data":{"label":"Input","nodeType":"Input",
                     "params":{"shape":"30,"}}},
            {"id":"2","type":"customNode","width":220,"height":80,
             "position":{"x":300,"y":180},
             "data":{"label":"Dense_1","nodeType":"Dense",
                     "params":{"units":128,"activation":"relu"}}},
            {"id":"3","type":"customNode","width":220,"height":80,
             "position":{"x":300,"y":310},
             "data":{"label":"Dropout_1","nodeType":"Dropout",
                     "params":{"rate":0.3}}},
            {"id":"4","type":"customNode","width":220,"height":80,
             "position":{"x":300,"y":440},
             "data":{"label":"Output","nodeType":"Output",
                     "params":{"units":1,"activation":"sigmoid"}}}
          ],
          "edges": [
            {"id":"e1-2","source":"1","target":"2","animated":True},
            {"id":"e2-3","source":"2","target":"3","animated":True},
            {"id":"e3-4","source":"3","target":"4","animated":True}
          ],
          "rationale": "Fallback architecture for binary classification."
        }
    
    # Store in BOTH places so nothing loses it
    cl.user_session.set("pending_graph", graph)
    cl.user_session.set("architect_graph", graph)
    
    return {
        "graph_architecture_json": graph,
        "is_architecture_modified": False,
        "architecture": f"CustomGraph | {graph.get('rationale', '')}"
    }


# ─────────────────────────────────────────────
# 4b.  HITL React Flow Pause Node
# ─────────────────────────────────────────────
def hitl_model_pause_node(state: AgentState) -> dict:
    """
    Pauses graph execution so Chainlit can render the React Flow visual editor.
    """
    chosen = interrupt({
        "action": "edit_architecture",
        "graph_json": state.get("graph_architecture_json", {})
    })
    
    # User clicks sync -> we obtain the new JSON structure
    # Wait, the interrupt return is whatever `resume()` gets called with. Let's just assume `chosen` is {"graph_architecture_json": {...}}
    return {
        "graph_architecture_json": chosen.get("graph_architecture_json", state.get("graph_architecture_json")),
        "is_architecture_modified": chosen.get("is_architecture_modified", True),
        "selected_architecture": "custom"
    }


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
    print(f"[kaggle_sourcer] Searching for: '{search_query}'")
    
    res = kaggle_search_tool.invoke({"query": search_query})
    return {"kaggle_results": res}

def huggingface_sourcer_node(state: AgentState) -> dict:
    """Searches HuggingFace for real datasets based on the user query."""
    llm = get_llm(temperature=0.0)
    search_query = _extract_keyword(state['user_query'], llm)
    print(f"[hf_sourcer] Searching for: '{search_query}'")
    
    res = hf_search_tool.invoke({"query": search_query})
    return {"hf_results": res}

def dataset_ranker_node(state: AgentState) -> dict:
    """Auto-resolves architecture selection and merges Kaggle & HuggingFace datasets."""
    import json
    kr = state.get("kaggle_results", "[]")
    
    try: kr_json = json.loads(kr)
    except: kr_json = []
    
    combined = [c for c in kr_json if "error" not in c]
    
    arch_full = "Custom Visual Graph" 
    
    if not combined:
        return {
            "architecture": arch_full,
            "dataset_options": [{
                "title": "⚠️ Search failed across both platforms",
                "ref":   "error/error",
                "url":   "https://kaggle.com",
                "source": "kaggle",
                "reason": "Search authentication or network failed."
            }]
        }
        
    llm = get_llm(temperature=0.0)
    prompt = textwrap.dedent(f"""
        User Problem: "{state['user_query']}"
        Available Datasets:
        {json.dumps(combined, indent=2)}
        
        Rank the top 3 best and most relevant datasets to solve the User Problem.
        Output exactly one valid JSON array of strings containing the 'ref' of the chosen datasets in order of preference.
        Example: ["owner/dataset1", "dataset2", "another-org/dataset3"]
    """).strip()
    
    response = llm.invoke(prompt)
    raw = _strip_code_fences(response.content)
    try: top_refs = json.loads(raw)
    except: top_refs = [combined[0]["ref"]] if combined else []
        
    dataset_options = []
    for ref in top_refs[:3]:
        for c in combined:
            if c["ref"] == ref:
                c["reason"] = f"Top selected dataset for '{state['user_query']}'"
                dataset_options.append(c)
                break
                
    if not dataset_options and combined:
        c = combined[0]
        c["reason"] = "Fallback allocation"
        dataset_options = [c]

    return {
        "architecture": arch_full,
        "dataset_options": dataset_options
    }



# ─────────────────────────────────────────────
# 4b.  HITL Pause Node
# ─────────────────────────────────────────────
def hitl_pause_node(state: AgentState) -> dict:
    """
    Pauses graph execution so Chainlit can present dataset buttons to the user.
    Resumes when the user clicks a button (injecting selected_dataset into state).
    """
    chosen = interrupt({
        "action":          "select_dataset",
        "dataset_options": state["dataset_options"],
        "architecture":    state["architecture"],
    })
    return {"selected_dataset": chosen}


# ─────────────────────────────────────────────
# 5.  Node 2 — Dataset Downloader (NEW — REAL)
# ─────────────────────────────────────────────
def dataset_downloader_node(state: AgentState) -> dict:
    """
    Downloads the selected Kaggle or HF dataset to disk.
    Stores the absolute local CSV path in state so the Engineer can use it directly.
    """
    ref = state["selected_dataset"]
    print(f"[downloader] Downloading dataset: {ref}")
    
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

    if csv_path.startswith("ERROR"):
        print(f"[downloader] Download failed: {csv_path}")
        # Fall back to the NASA C-MAPSS data already in the workspace
        import pathlib
        fallback = str(pathlib.Path("train_FD001.txt").absolute())
        print(f"[downloader] Using NASA C-MAPSS fallback: {fallback}")
        csv_path = fallback

    print(f"[downloader] ✅ CSV ready at: {csv_path}")
    return {"dataset_csv_path": csv_path}


# ─────────────────────────────────────────────
# 5b.  Node — EDA Analyzer (Sprint 5)
# ─────────────────────────────────────────────
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
        return {"eda_data": {}, "eda_narration": "Data source not found."}

    # ── Clear previous run ──────────────────────────────────────────────────
    with _eda_lock:
        _eda_steps[session_id] = []
        _eda_done[session_id]  = False

    print(f"[eda] Profiling dataset: {csv_path}")

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
        print(f"[eda] CSV read error: {e}")
        _eda_emit(session_id, "load", "Loading CSV", status="error",
                  detail=str(e), pct=0)
        with _eda_lock:
            _eda_done[session_id] = True
        return {"eda_data": {}, "eda_narration": f"Error reading data: {e}"}

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

    return {
        "eda_data": {
            "columns":       df.columns.tolist(),
            "row_count":     len(df),
            "col_count":     len(df.columns),
            "distributions": distributions,
            "categoricals":  categoricals,
            "correlation":   corr_matrix,
            "missing":       missing_report,
            "outliers":      outlier_report,
            "top_corrs":     top_corrs[:10],
        },
        "eda_narration": narration
    }


# ─────────────────────────────────────────────
# 5c.  HITL — EDA Dashboard Pause
# ─────────────────────────────────────────────
def hitl_eda_pause_node(state: AgentState) -> dict:
    """
    Pauses so the user can explore the interactive EDA dashboard.
    """
    interrupt({
        "action":    "show_eda_dashboard",
        "eda_data":  state["eda_data"],
        "narration": state["eda_narration"]
    })
    return {}


# ─────────────────────────────────────────────
# 5d.  HITL — Execution Choice Node (NEW)
# ─────────────────────────────────────────────
def execution_choice_node(state: AgentState) -> dict:
    """
    Pauses graph execution so the user can choose between local or cloud execution.
    """
    choice = interrupt({
        "action": "select_execution_mode",
        "options": ["local", "cloud"],
        "architecture": state["architecture"].split("|")[0].strip()
    })
    return {"execution_mode": choice}


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
            print(f"[syntax-fix] Attempt {attempt+1}/{max_attempts} — SyntaxError at line {exc.lineno}: {exc.msg}")
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
                print(f"[syntax-fix] LLM call failed: {e}")
                break
    return code  # return best effort


# ─────────────────────────────────────────────
# 6b. Engineer Agent
# ─────────────────────────────────────────────
def engineer_node(state: AgentState) -> dict:
    import json
    llm = get_llm(temperature=0.2)
    final_graph = state.get("final_graph", {})
    space = state.get("hpt_search_space", {})
    tc = state.get("training_config") or {}

    # Human-defined hyperparameters (with defaults)
    epochs      = int(tc.get("epochs", 50))
    test_size   = float(tc.get("test_size", 0.2))
    batch_size  = int(tc.get("batch_size", 64))
    optimizer   = str(tc.get("optimizer", "adam")).lower()
    lr          = float(tc.get("lr", 0.001))
    early_stop  = bool(tc.get("early_stop", True))
    dropout     = bool(tc.get("dropout", True))
    class_wts   = bool(tc.get("class_weights", True))
    hpt_trials  = int(tc.get("hpt_trials", 15))
    seed        = int(tc.get("seed", 42))
    patience    = max(10, epochs // 5)

    if not final_graph.get("nodes"):
        return {**state, "generated_code": None}

    # NOTE: Use NO single-quotes around optimizer value to prevent LLM from
    # echoing the quote character into generated code (causes SyntaxError).
    sys_prompt = textwrap.dedent(f"""
        You are an elite PyTorch Engineer.
        Generate a single monolithic Python script for hyperparameter tuning and training.
        Output ONLY raw Python — no markdown fences, no triple-backticks, no explanation.

        GRAPH TOPOLOGY JSON: {json.dumps(final_graph)}
        OPTUNA SEARCH SPACE: {json.dumps(space)}

        EXACT HYPERPARAMETER VALUES — hard-code these into the script directly:
        NUM_EPOCHS   = {epochs}
        TEST_SIZE    = {test_size}
        BATCH_SIZE   = {batch_size}
        OPTIMIZER    = "{optimizer}"
        LR           = {lr}
        EARLY_STOP   = {str(early_stop).lower()}
        DROPOUT      = {str(dropout).lower()}
        CLASS_WEIGHTS= {str(class_wts).lower()}
        HPT_TRIALS   = {hpt_trials}
        SEED         = {seed}

        REQUIREMENTS — follow exactly:
        1. Load CSV from "{state.get('dataset_csv_path', 'train.csv')}" (use the variable CSV_PATH = "<path>").
        2. Preprocessing IN THIS ORDER:
           a. df = pd.read_csv(CSV_PATH)
           b. df = df.drop(columns=[c for c in df.columns if df[c].isnull().mean() > 0.5])
           c. df = df.select_dtypes(include=["number"])
           d. df = df.dropna()
           e. if len(df) == 0: raise RuntimeError("DataFrame is empty after preprocessing")
           f. X = df.iloc[:, :-1].values.astype(np.float32)
           g. y = df.iloc[:, -1].values
           h. X_train, X_val, y_train, y_val = train_test_split(X, y, test_size={test_size}, random_state={seed})
        3. Build PyTorch model matching the graph topology. Set input_dim = X_train.shape[1].
        4. Run Optuna (n_trials={hpt_trials}, seed={seed}) to find the best hyperparameters. Train for EXACTLY 3 epochs during Optuna trials to save time. DO NOT print any epoch_metric during Optuna. Per trial print:
           print(json.dumps({{"type":"hpt_trial","trial":t.number,"total":{hpt_trials},"params":t.params,"value":float(v),"status":"complete","best_so_far":float(s)}}), flush=True)
        5. After HPT print:
           print(json.dumps({{"type":"hpt_complete","best_params":best_params,"best_value":float(best_val),"total_trials":{hpt_trials}}}), flush=True)
        6. VERY IMPORTANT: You MUST train a FINAL model using the `best_params` from Optuna. Train this final model for EXACTLY NUM_EPOCHS epochs outside of Optuna. You MUST use exactly this loop: `for epoch in range(NUM_EPOCHS):`. DO NOT USE Optuna inside this final loop!
           {f'Use early stopping with patience={patience} during final training.' if early_stop else ''}
        7. During the FINAL training loop ONLY, each epoch print:
           print(json.dumps({{"type":"epoch_metric","epoch":epoch+1,"loss":float(loss),"val_loss":float(vloss),"acc":float(acc),"val_acc":float(vacc)}}), flush=True)
        8. Do NOT use any f-strings or string formatting that references undefined variables.
        9. Every string literal must be properly closed on the same line.
    """).strip()

    response = llm.invoke(sys_prompt)
    generated_code = _strip_code_fences(response.content)
    # Validate & auto-fix syntax before storing
    generated_code = _validate_and_fix_syntax(generated_code, llm)
    return {"generated_code": generated_code, "retry_count": 0, "training_logs": "", "metrics": []}

# ─────────────────────────────────────────────
# 6c. Groq Loopfixer
# ─────────────────────────────────────────────
def groq_loopfixer_node(state: AgentState) -> dict:
    script = state.get("generated_code", "")
    if not script: return {}

    llm = get_llm(temperature=0.0)
    prompt = textwrap.dedent(f"""
    You are a senior ML engineer performing a pre-flight code review.
    Fix ALL of the following issues in the script:
    1. Any column that cannot be cast to float32 (datetime, string, object) — add df.select_dtypes(include=["number"])
    2. Any hardcoded input_dim that doesn't match the actual CSV columns — replace with: input_dim = X_train.shape[1]
    3. Missing train/val split — add sklearn train_test_split.
    4. Missing epoch metric emission — every epoch of the FINAL model training loop (NOT Optuna trials) MUST print JSON line epoch_metric exactly.
    5. Class imbalance — if binary/multi classification, compute and apply class_weights.
    6. NaN in target or features — MANDATORY preprocessing before split:
         df = df.drop(columns=[c for c in df.columns if df[c].isnull().mean() > 0.5])
         df = df.select_dtypes(include=["number"])
         df = df.dropna()
         if len(df) == 0: raise RuntimeError("DataFrame empty after preprocessing")
    7. Ensure y is cast correctly: long (torch.long) for classification, float32 for regression.
    8. Fix any syntax errors — every string literal must be properly closed, no unterminated strings.
    9. Remove any markdown fences or triple-backtick lines.

    Return ONLY the corrected raw Python script. No markdown fences. No explanation.

    SCRIPT TO REVIEW:
    {script}
    """).strip()

    res = llm.invoke(prompt)
    fixed = _strip_code_fences(res.content)
    # Always validate syntax after loopfixer repairs
    fixed = _validate_and_fix_syntax(fixed, llm)

    return {"groq_fixed_code": fixed}

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
    script = state.get("groq_fixed_code", "") or state.get("generated_code", "")
    if not script or not script.strip():
        return {
            **state,
            "training_logs": "ERROR: Engineer agent did not produce code.",
            "metrics": []
        }

    # — PRE-FLIGHT: AST syntax check before writing to disk —
    llm_for_fix = get_llm(temperature=0.0)
    try:
        ast.parse(script)
    except SyntaxError as se:
        print(f"[sandbox] SyntaxError detected pre-flight at line {se.lineno}: {se.msg} — auto-repairing...")
        script = _validate_and_fix_syntax(script, llm_for_fix)
        try:
            ast.parse(script)
            print("[sandbox] Pre-flight syntax repair succeeded.")
        except SyntaxError as se2:
            err = f"FATAL SyntaxError after repair: {se2.msg} at line {se2.lineno}"
            print(f"[sandbox] {err}")
            return {**state, "training_logs": err, "metrics": []}

    print("[sandbox] 💻 Local mode selected. Subprocess streaming enabled...")
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

        metrics_list = []
        while True:
            line = await process.stdout.readline()
            if not line:
                break
            
            decoded_line = line.decode('utf-8', errors='ignore').strip()
            stdout_lines.append(decoded_line)
            
            try:
                j = json.loads(decoded_line)
                t = j.get("type", "")
                if t == "hpt_trial":
                    hpt_state["current_trial"] = j.get("trial", 0)
                    hpt_state["total_trials"] = j.get("total", 15)
                    hpt_state["best_value"] = j.get("best_so_far", 0.0)
                    hpt_state["current_params"] = j.get("params", {})
                    hpt_state["trials"].append(j)
                    hpt_state["logs"] = stdout_lines[-20:]
                elif t == "hpt_complete":
                    hpt_state["status"] = "complete"
                    hpt_state["best_params"] = j.get("best_params", {})
                    training_state["best_params"] = j.get("best_params", {})
                elif t == "epoch_metric":
                    training_state["current_epoch"] = j.get("epoch", 0)
                    metrics_list.append(j)
                    training_state["metrics"] = metrics_list
                    training_state["architecture"] = state.get("architecture_desc", "")
                    training_state["logs"] = stdout_lines[-20:]
                    
                    if len(metrics_list) % 5 == 0:
                        asyncio.create_task(fetch_commentary(metrics_list[-5:]))
            except json.JSONDecodeError:
                if hpt_state["status"] != "complete": 
                    hpt_state["logs"] = stdout_lines[-20:]
                training_state["logs"] = stdout_lines[-20:]
        
        await process.wait()
        logs = "\\n".join(stdout_lines[-5000:])
    except Exception as exc:
        logs = f"ERROR: Subprocess failed — {exc}"
    finally:
        try: os.remove(tmp_path)
        except: pass

    return {"training_logs": logs, "metrics": metrics_list if 'metrics_list' in locals() else []}


# ─────────────────────────────────────────────
# 8b.  Node — Debugger (Auto-Healing)
# ─────────────────────────────────────────────
def debugger_node(state: AgentState) -> dict:
    """
    Reads the Traceback from training_logs and the crashed generated_code.
    Ask the LLM to patch it.
    """
    llm = get_llm(temperature=0.0)
    
    prompt = textwrap.dedent(f"""
        You are an elite autonomous debugging agent.
        The previous training script crashed with the following error:
        
        === TRACEBACK ===
        {state.get("training_logs", "")}
        
        === CODE ===
        {state.get("generated_code", "")}
        
        === REQUIRED CONFIGURATION (DO NOT CHANGE) ===
        {json.dumps(state.get("training_config", {}), indent=2)}
        
        Fix the python script to resolve this error entirely. 
        MANDATORY: 
        1. Maintain all imports and the data loading logic.
        2. Do NOT change the user's desired hyperparameters (Epochs, Optimizer, etc.) unless specifically requested by the error (e.g. out of memory).
        3. Do NOT use markdown code fences. Output ONLY the raw Python code.
    """).strip()
    
    response = llm.invoke(prompt)
    new_code = _strip_code_fences(response.content)
    
    return {
        "generated_code": new_code,
        "retry_count": state.get("retry_count", 0) + 1,
        "training_logs": "" # Reset logs for the retry
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
    print(f"[comparator] 📡 ArXiv Search Query: {search_query}")
    
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
        print("[evaluator] ✅ PDF report generated successfully.")
    except Exception as e:
        print(f"[evaluator] ❌ PDF generation failed: {e}")

    return {"final_report": report_content}

def check_execution_success(state: AgentState) -> str:
    """
    Checks if local execution produced a traceback, routing to debugger if so. 
    Respects a maximum of 3 retries.
    """
    logs = state.get("training_logs", "")
    retries = state.get("retry_count", 0)
    mode = state.get("execution_mode", "local")
    
    if mode == "local" and ("Traceback" in logs or "Error" in logs or "Exception" in logs):
        if retries < 3:
            return "debugger"
            
    return "arxiv_comparator"

def check_engineer_code(state: AgentState) -> str:
    generated_code = state.get("generated_code")
    if not generated_code or not generated_code.strip():
        return "hitl_model_pause"
    return "execution_choice"

# ─────────────────────────────────────────────
# 9.  Graph Assembly
# ─────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)
    graph.add_node("architect",          architect_node)
    graph.add_node("hitl_model_pause",   hitl_model_pause_node)
    graph.add_node("kaggle_sourcer",     kaggle_sourcer_node)
    graph.add_node("dataset_ranker",     dataset_ranker_node)
    graph.add_node("hitl_pause",         hitl_pause_node)
    graph.add_node("dataset_downloader", dataset_downloader_node)
    graph.add_node("eda_analyzer",       eda_analyzer_node)
    graph.add_node("hitl_eda_pause",     hitl_eda_pause_node)
    graph.add_node("execution_choice",   execution_choice_node)
    graph.add_node("hpt",                hpt_node)
    graph.add_node("engineer",           engineer_node)
    graph.add_node("groq_loopfixer",     groq_loopfixer_node)
    graph.add_node("execution_sandbox",  execution_sandbox_node)
    graph.add_node("debugger",           debugger_node)
    graph.add_node("arxiv_comparator",   arxiv_comparator_node)
    graph.add_node("evaluator",          evaluator_node)

    graph.set_entry_point("architect")
    graph.add_edge("architect",          "hitl_model_pause")
    graph.add_edge("hitl_model_pause",   "kaggle_sourcer")
    graph.add_edge("kaggle_sourcer",     "dataset_ranker")
    graph.add_edge("dataset_ranker",     "hitl_pause")
    graph.add_edge("hitl_pause",         "dataset_downloader")
    graph.add_edge("dataset_downloader", "eda_analyzer")
    graph.add_edge("eda_analyzer",       "hitl_eda_pause")
    graph.add_edge("hitl_eda_pause",     "execution_choice")
    graph.add_edge("execution_choice",   "hpt")
    graph.add_edge("hpt",                "engineer")
    
    graph.add_conditional_edges(
        "engineer",
        check_engineer_code,
        {"execution_choice": "groq_loopfixer", "hitl_model_pause": "hitl_model_pause"}
    )
    
    graph.add_edge("groq_loopfixer",     "execution_sandbox")
    
    graph.add_conditional_edges("execution_sandbox", check_execution_success, {"debugger": "debugger", "arxiv_comparator": "arxiv_comparator"})
    graph.add_edge("debugger",           "execution_sandbox")
    graph.add_edge("arxiv_comparator",   "evaluator")
    graph.add_edge("evaluator",          END)

    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()

    return graph.compile(
        checkpointer=memory,
        interrupt_before=["hitl_model_pause", "hitl_pause", "execution_choice"],
    )
