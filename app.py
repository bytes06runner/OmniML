"""
app.py - AnomaLLM v3 Chainlit Frontend
=======================================
Provides a dynamic chat interface that:
  • Streams LangGraph node progress via cl.Step
  • Handles the HITL interrupt by surfacing 3 REAL Kaggle dataset buttons
  • Resumes graph execution with the user's choice
  • NEW: Handles HITL choice for Local vs Cloud (Kaggle GPU) execution
  • Shows a "Downloading Dataset…" step for the new dataset_downloader node
  • Streams the final Markdown report

Author: AnomaLLM v3 / Antigravity
"""

import os
import uuid
import json
import time
import tools
import chainlit as cl
from groq import Groq
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from graph import build_graph
from anomallm.persistence import SQLiteDataLayer
from anomallm.plugins import PluginRegistry

from chainlit.server import app as fastapi_app

# ─────────────────────────────────────────────────────────────────────────────
# Global Cache for Iframe Sync
# ─────────────────────────────────────────────────────────────────────────────
_session_graphs:          dict = {}
_session_problems:        dict = {}
_session_training_configs: dict = {}   # session_id -> training config dict
_session_threads:          dict = {}   # session_id -> thread_id string
_session_runs:             dict = {}   # session_id -> run_id string
_session_run_manifests:    dict = {}   # session_id -> manifest dict
_session_eda_payloads:     dict = {}   # session_id -> {"eda_data": ..., "eda_narration": ...}
_session_execution_state:  dict = {}   # session_id -> {"started": bool, "completed": bool}
_groq_health_cache:        dict = {"checked_at": 0.0, "ok": None, "message": ""}
_inline_view_registry:     dict = {}   # anchor_id -> descriptor
_last_dataset_download_result: dict = {}

_pipeline_stages: dict = {}

def set_stage(stage_id: str, status: str):
    _pipeline_stages[stage_id] = status


def default_training_config() -> dict:
    return {
        "epochs": 50,
        "test_size": 0.2,
        "batch_size": 64,
        "optimizer": "adam",
        "lr": 0.001,
        "early_stop": True,
        "dropout": True,
        "class_weights": True,
        "shuffle": True,
        "hpt_trials": 15,
        "seed": 42,
        "compliance_modes": ["eu_ai_act", "fda_samd", "soc2"],
        "fairness_config": {
            "protected_attributes": [],
            "disparity_threshold": 0.05,
            "minimum_group_size": 25,
            "backend": "fairlearn",
        },
        "benchmark_config": {
            "mode": "prefer_live_then_cache",
            "cache_ttl_days": 7,
            "source_timeout_seconds": 10,
        },
        "plugin_config": {
            "enabled_plugins": [],
            "plugin_overrides": {},
        },
    }


def _merge_training_config(cfg: dict | None) -> dict:
    defaults = default_training_config()
    merged = {**defaults, **(cfg or {})}
    merged["fairness_config"] = {**defaults["fairness_config"], **((cfg or {}).get("fairness_config") or {})}
    merged["benchmark_config"] = {**defaults["benchmark_config"], **((cfg or {}).get("benchmark_config") or {})}
    merged["plugin_config"] = {**defaults["plugin_config"], **((cfg or {}).get("plugin_config") or {})}
    return merged


def _normalize_manifest_payload(payload: dict | None) -> dict | None:
    if not isinstance(payload, dict):
        return None
    run_manifest = payload.get("run_manifest") if isinstance(payload.get("run_manifest"), dict) else payload
    if not isinstance(run_manifest, dict):
        return None
    return {
        "run_id": run_manifest.get("run_id"),
        "paths": run_manifest.get("paths", {}) or {},
        "artifact_refs": run_manifest.get("artifact_refs", []) or [],
        "run_manifest": run_manifest,
        "bundle": payload if isinstance(payload.get("run_manifest"), dict) else {"run_manifest": run_manifest},
    }


def _compliance_downloads(run_id: str | None, template_id: str) -> dict:
    safe_run_id = run_id or "unknown"
    return {
        "markdown": f"/dl-run-artifact/{safe_run_id}/report_markdown/{template_id}.md",
        "html": f"/dl-run-artifact/{safe_run_id}/report_html/{template_id}.html",
        "pdf": f"/dl-run-artifact/{safe_run_id}/report_pdf/{template_id}.pdf",
    }


def _update_session_run_state(session_id: str, node_state: dict | None):
    if not node_state:
        return
    manifest = _normalize_manifest_payload(node_state)
    if manifest:
        _session_run_manifests[session_id] = manifest
        run_id = manifest.get("run_id")
        if run_id:
            _session_runs[session_id] = run_id


def _load_manifest_for_session(session_id: str) -> dict | None:
    manifest = _session_run_manifests.get(session_id)
    if manifest:
        return manifest
    run_id = _session_runs.get(session_id)
    if not run_id:
        return None
    manifest_path = os.path.join(os.getcwd(), "runs", run_id, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = _normalize_manifest_payload(json.load(handle))
    if manifest:
        _session_run_manifests[session_id] = manifest
    return manifest


async def send_status_message(title: str, body: str, actions: list | None = None):
    content = f"## {title}\n\n{body}" if title else body
    await cl.Message(content=content, actions=actions or []).send()


def _register_inline_view(session_id: str, view: str, url: str, title: str) -> str:
    anchor_id = uuid.uuid4().hex
    _inline_view_registry[anchor_id] = {
        "type": "inline_view",
        "anchor_id": anchor_id,
        "session_id": session_id or "default",
        "view": view,
        "title": title,
        "url": url,
    }
    return anchor_id


async def send_inline_view(
    session_id: str,
    view: str,
    url: str,
    title: str,
    body: str,
    actions: list | None = None,
):
    anchor_id = _register_inline_view(session_id, view, url, title)
    separator = "&" if "?" in url else "?"
    hydration_url = f"{url}{separator}omniml_anchor={anchor_id}"
    fallback_link = f"[Open {title}]({hydration_url})"
    content = (
        f"## {title}\n\n"
        f"{body}\n\n"
        f"If the embedded view does not load, use {fallback_link}."
    )
    await cl.Message(content=content, actions=actions or []).send()


def _check_groq_health(force: bool = False) -> tuple[bool, str]:
    now = time.time()
    if (
        not force
        and _groq_health_cache["ok"] is not None
        and now - _groq_health_cache["checked_at"] < 60
    ):
        return bool(_groq_health_cache["ok"]), str(_groq_health_cache["message"])

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        message = (
            "Groq is not configured. Set GROQ_API_KEY in .env and restart OmniML."
        )
        _groq_health_cache.update({"checked_at": now, "ok": False, "message": message})
        return False, message

    try:
        client = Groq(api_key=api_key)
        client.chat.completions.create(
            model=os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b"),
            messages=[{"role": "user", "content": "ping"}],
            temperature=0,
            max_tokens=4,
        )
        message = "Groq authentication check passed."
        _groq_health_cache.update({"checked_at": now, "ok": True, "message": message})
        return True, message
    except Exception as exc:
        error_text = str(exc)
        if "invalid_api_key" in error_text or "401" in error_text:
            message = (
                "Groq rejected GROQ_API_KEY with 401 invalid_api_key. "
                "Replace the key in .env, save it without quotes, and restart OmniML."
            )
        else:
            message = f"Groq preflight failed: {error_text}"
        _groq_health_cache.update({"checked_at": now, "ok": False, "message": message})
        return False, message

@fastapi_app.get("/pipeline-status")
async def pipeline_status():
    return JSONResponse({"stages": _pipeline_stages})

@fastapi_app.post("/sync-graph")
async def sync_graph(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "default")
    nodes = body.get("nodes", [])
    edges = body.get("edges", [])
    
    if not nodes:
        return JSONResponse({"ok": False, "error": "Empty graph"}, status_code=400)
    
    _session_graphs[session_id] = {"nodes": nodes, "edges": edges, "synced": True}
    return JSONResponse({"ok": True, "nodes": len(nodes), "edges": len(edges)})

@fastapi_app.get("/hpt-status")
async def get_hpt_status():
    from graph import hpt_state
    return JSONResponse(hpt_state)

@fastapi_app.get("/training-status")
async def get_training_status():
    from graph import training_state
    return JSONResponse(training_state)

@fastapi_app.get("/deploy-status")
async def deployment_status(session_id: str = "default"):
    """Return export artifact status for the deployment dashboard."""
    manifest = _load_manifest_for_session(session_id)
    export_dir = os.path.join(os.getcwd(), "exports")
    if manifest:
        export_dir = manifest.get("paths", {}).get("exports", export_dir)
    meta_path  = os.path.join(export_dir, "model_meta.json")

    if not os.path.exists(meta_path):
        return JSONResponse({"ready": False, "meta": None, "files": {}})

    with open(meta_path) as f:
        meta = json.load(f)

    filenames = [
        "model.pt", "model.onnx", "model_scripted.pt",
        "model_meta.json", "serve_api.py", "Dockerfile", "requirements.txt"
    ]
    files = {}
    for fn in filenames:
        p = os.path.join(export_dir, fn)
        if os.path.exists(p):
            files[fn] = {"exists": True, "size": os.path.getsize(p)}
        else:
            files[fn] = {"exists": False, "size": 0}

    return JSONResponse({"ready": True, "meta": meta, "files": files, "run_id": manifest.get("run_id") if manifest else None})

@fastapi_app.get("/dl-artifact/{filename}")
async def download_artifact(filename: str):
    """Serve exported model artifacts for download."""
    import os
    safe_names = {
        "model.pt", "model.onnx", "model.onnx.data", "model_scripted.pt",
        "model_meta.json", "serve_api.py", "Dockerfile", "requirements.txt"
    }
    if filename not in safe_names:
        return JSONResponse({"error": "Invalid filename"}, status_code=400)

    path = os.path.join(os.getcwd(), "exports", filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(path, filename=filename)


@fastapi_app.get("/plugin-catalog")
async def plugin_catalog():
    registry = PluginRegistry()
    catalog = [manifest.model_dump(mode="json") for manifest in registry.discover()]
    return JSONResponse({"plugins": catalog, "empty": len(catalog) == 0})


@fastapi_app.get("/run-artifacts")
async def run_artifacts(session_id: str = "default"):
    manifest = _load_manifest_for_session(session_id)
    if not manifest:
        return JSONResponse({"ok": False, "error": "No run manifest found for session."}, status_code=404)
    artifact_refs = manifest.get("artifact_refs", [])
    grouped = {}
    for artifact in artifact_refs:
        grouped.setdefault(artifact.get("kind", "unknown"), []).append(artifact)
    return JSONResponse({
        "ok": True,
        "run_id": manifest.get("run_id"),
        "artifact_refs": artifact_refs,
        "grouped": grouped,
        "paths": manifest.get("paths", {}),
    })


@fastapi_app.get("/compliance-status")
async def compliance_status(session_id: str = "default"):
    manifest = _load_manifest_for_session(session_id)
    if not manifest:
        return JSONResponse({"ok": False, "error": "No run manifest found for session."}, status_code=404)
    reports_dir = manifest.get("paths", {}).get("reports", "")
    reports = []
    bundle_reports = manifest.get("bundle", {}).get("compliance_artifacts", []) or []
    if bundle_reports:
        for report in bundle_reports:
            template_id = report.get("template_id", "unknown")
            markdown_path = report.get("markdown_path")
            completeness = (report.get("validation") or {}).get("completeness", "unknown")
            reports.append({
                "template_id": template_id,
                "markdown_path": markdown_path,
                "completeness": completeness,
                "missing_required_count": len((report.get("validation") or {}).get("missing_required", []) or []),
                "downloads": _compliance_downloads(manifest.get("run_id"), template_id),
            })
    elif reports_dir and os.path.isdir(reports_dir):
        for filename in os.listdir(reports_dir):
            if not filename.endswith(".md"):
                continue
            path = os.path.join(reports_dir, filename)
            with open(path, "r", encoding="utf-8") as handle:
                content = handle.read()
            reports.append({
                "template_id": filename[:-3],
                "markdown_path": path,
                "completeness": _extract_report_completeness(content),
                "missing_required_count": content.count("- ") if "## Missing Required Evidence" in content else 0,
                "downloads": _compliance_downloads(manifest.get("run_id"), filename[:-3]),
            })
    return JSONResponse({"ok": True, "run_id": manifest.get("run_id"), "reports": reports})


@fastapi_app.get("/benchmark-status")
async def benchmark_status(session_id: str = "default"):
    manifest = _load_manifest_for_session(session_id)
    if not manifest:
        return JSONResponse({"ok": False, "error": "No run manifest found for session."}, status_code=404)
    artifact_path = os.path.join(manifest.get("paths", {}).get("artifacts", ""), "benchmark_summary.json")
    if not os.path.exists(artifact_path):
        return JSONResponse({"ok": False, "error": "No benchmark artifact found."}, status_code=404)
    with open(artifact_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return JSONResponse({"ok": True, "run_id": manifest.get("run_id"), "benchmark": payload})


@fastapi_app.get("/dl-run-artifact/{run_id}/{kind}/{filename}")
async def download_run_artifact(run_id: str, kind: str, filename: str):
    manifest_path = os.path.join(os.getcwd(), "runs", run_id, "manifest.json")
    if not os.path.exists(manifest_path):
        return JSONResponse({"error": "Run not found"}, status_code=404)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = _normalize_manifest_payload(json.load(handle))
    if not manifest:
        return JSONResponse({"error": "Run manifest invalid"}, status_code=404)
    matches = [artifact for artifact in manifest.get("artifact_refs", []) if artifact.get("kind") == kind and os.path.basename(artifact.get("path", "")) == filename]
    if not matches:
        return JSONResponse({"error": "Artifact not found"}, status_code=404)
    path = matches[0]["path"]
    if not os.path.exists(path):
        return JSONResponse({"error": "Artifact path missing on disk"}, status_code=404)
    return FileResponse(path, filename=filename)


def _extract_report_completeness(markdown: str) -> str:
    for line in markdown.splitlines():
        if line.startswith("Completeness:"):
            return line.split("**")[-2] if "**" in line else line.split(":", 1)[1].strip()
    return "unknown"

@fastapi_app.get("/get-architect-graph")
async def get_architect_graph(session_id: str = "default"):
    graph = _session_graphs.get(session_id)
    if not graph or not graph.get("nodes"):
        return JSONResponse({
            "ok": False, 
            "error": "No graph found"
        }, status_code=404)
    return JSONResponse({"ok": True, "graph": graph})

@fastapi_app.get("/eda-progress")
async def eda_progress(session_id: str = "default"):
    """SSE endpoint - streams EDA step events to the live progress dashboard."""
    import asyncio, json as _json
    from graph import _eda_steps, _eda_done, _eda_lock

    async def event_stream():
        sent = 0
        while True:
            with _eda_lock:
                steps  = list(_eda_steps.get(session_id, []))
                done   = _eda_done.get(session_id, False)
            # Stream only new events
            for step in steps[sent:]:
                data = _json.dumps(step)
                yield f"data: {data}\n\n"
                sent += 1
            if done and sent >= len(steps):
                yield "data: {\"type\":\"done\"}\n\n"
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@fastapi_app.post("/training-config-store")
async def store_training_config(request: Request, session_id: str = "default"):
    """Store human-defined training hyperparameters for the given session."""
    body = await request.json()
    merged = _merge_training_config(body)
    _session_training_configs[session_id] = merged
    print(f"[training-config] Stored for session={session_id}: {merged}")
    return JSONResponse({"ok": True, "session_id": session_id, "config": merged})

@fastapi_app.get("/training-config-get")
async def get_training_config(session_id: str = "default"):
    cfg = _merge_training_config(_session_training_configs.get(session_id))
    _session_training_configs.setdefault(session_id, cfg)
    return JSONResponse({"ok": True, "config": cfg})


@fastapi_app.get("/inline-view-state")
async def inline_view_state(anchor_id: str):
    descriptor = _inline_view_registry.get(anchor_id)
    if not descriptor:
        return JSONResponse({"ok": False, "error": "Inline view not found"}, status_code=404)
    return JSONResponse({"ok": True, **descriptor})


@fastapi_app.get("/eda-data")
async def get_eda_data(session_id: str = "default"):
    payload = _session_eda_payloads.get(session_id)
    if not payload:
        return JSONResponse({"ok": False, "error": "EDA payload not found"}, status_code=404)
    return JSONResponse({"ok": True, **payload})


@fastapi_app.get("/llm-status")
async def llm_status():
    ok, message = _check_groq_health(force=True)
    status_code = 200 if ok else 503
    return JSONResponse({"ok": ok, "provider": "groq", "message": message}, status_code=status_code)


@fastapi_app.get("/runtime-diagnostics")
async def runtime_diagnostics():
    groq_ok, groq_message = _check_groq_health(force=True)
    kaggle_ok, _, kaggle_message = tools._kaggle_preflight()
    return JSONResponse(
        {
            "ok": True,
            "groq": {"ok": groq_ok, "message": groq_message},
            "kaggle": {
                "ok": kaggle_ok,
                "cli_path": tools._resolve_kaggle_cli_path(),
                "credentials_present": bool(os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")),
                "message": kaggle_message,
            },
            "last_dataset_download_result": _last_dataset_download_result,
        }
    )

@fastapi_app.get("/suggest-architecture")
async def suggest_architecture(session_id: str = "default"):
    try:
        groq_ok, groq_message = _check_groq_health(force=True)
        if not groq_ok:
            return JSONResponse(
                {
                    "ok": False,
                    "error": groq_message,
                    "graph": None,
                },
                status_code=503,
            )

        problem = _session_problems.get(session_id, "binary classification")
        dataset_info = {}
        try:
            if getattr(cl.context, "session", None):
                dataset_info = cl.user_session.get("dataset_metadata") or {}
        except Exception:
            pass
        n_features = dataset_info.get("columns", "unknown")
        n_classes = dataset_info.get("n_classes", 2)
        dataset_size = dataset_info.get("rows", "unknown")
        
        prompt = f"""
You are a senior deep learning architect. The user's problem:
"{problem}"

Dataset info:
- Features: {n_features}
- Classes: {n_classes}
- Rows: {dataset_size}

Generate a complete neural network architecture as a JSON node 
graph for React Flow. Rules you MUST follow:
1. ALL hidden layers must use ReLU activation
2. Output layer must use Sigmoid for binary, Softmax for multi-class,
   Linear for regression
3. Use Dropout(0.3) after every Dense layer with >64 units
4. Use BatchNorm1d after the first Dense layer
5. Scale depth and width to dataset size:
   - Small (<5k rows): 2 Dense layers, max 128 units
   - Medium (5k-50k): 3 Dense layers, max 256 units  
   - Large (>50k): 4 Dense layers, max 512 units
6. Node positions must be vertical top-to-bottom, x=300 for all,
   y increments of 130 starting from y=50
7. Every node must have: width=220, height=80

Return ONLY valid JSON, no markdown, no explanation:
{{
  "nodes": [
    {{"id":"1","type":"customNode","width":220,"height":80,
      "position":{{"x":300,"y":50}},
      "data":{{"label":"Input","nodeType":"Input",
               "params":{{"shape":"{n_features},"}}}}}},
    {{"id":"2","type":"customNode","width":220,"height":80,
      "position":{{"x":300,"y":180}},
      "data":{{"label":"Dense_1","nodeType":"Dense",
               "params":{{"units":128,"activation":"relu"}}}}}},
    ...output node last with sigmoid/softmax/linear...
  ],
  "edges": [
    {{"id":"e1-2","source":"1","target":"2","animated":true}},
    ...
  ],
  "rationale": "One sentence why this architecture fits the problem."
}}
"""
        
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000,
        )
        
        raw = response.choices[0].message.content.strip()
        
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        
        graph = json.loads(raw)
        return JSONResponse({
            "ok": True,
            "graph": graph,
            "rationale": graph.get("rationale", "")
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({
            "ok": False,
            "error": str(e),
            "graph": {
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
              "rationale": "Fallback architecture due to endpoint failure."
            }
        })

# ─────────────────────────────────────────────────────────────────────────────
# Global Graph Configuration
# ─────────────────────────────────────────────────────────────────────────────
cl.data_layer = SQLiteDataLayer("sqlite:///database.db")
_graph = build_graph()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _node_label(node_name: str) -> str:
    return {
        "architect":          "🧠 Architecting Graph",
        "dataset_ranker":     "🔍 Ranking API Datasets",
        "dataset_downloader": "⬇️  Downloading Real Data",
        "hpt_node":           "🎛️  Tuning Hyperparameters",
        "engineer":           "⚙️  Engineering Optuna Code",
        "groq_loopfixer":     "🛠️  Code Review & Self-Correction",
        "execution_choice":   "🤔 Choosing Execute Environment",
        "execution_sandbox":  "🚀 Executing Live Process",
        "evaluator":          "📊 Generating Final Report",
    }.get(node_name, f"🔄 {node_name}")
    
def render_dataset_card(d: dict, source: str) -> str:
    icon = "🇰" if source == "kaggle" else "🤗"
    badge = "Kaggle" if source == "kaggle" else "HuggingFace"
    
    meta_parts = []
    if d.get("rows"):       meta_parts.append(f"**{d['rows']:,}** rows")
    if d.get("columns"):    meta_parts.append(f"**{d['columns']}** cols")
    if d.get("size_mb"):    meta_parts.append(f"**{d['size_mb']}** MB")
    if d.get("downloads"):  meta_parts.append(f"**{d['downloads']:,}** DLs")
    if d.get("votes"):      meta_parts.append(f"**{d['votes']}** votes")
    if d.get("last_updated"): meta_parts.append(f"updated **{d['last_updated']}**")
    if d.get("license"):    meta_parts.append(f"license: {d['license']}")
    
    meta_line = " · ".join(meta_parts)
    link = d.get("url", "#")
    name = d.get("title") or d.get("dataset_id", "Unknown")
    desc = d.get("description", "")
    
    return f"{icon} **[{name}]({link})** `{badge}`\n{desc}\n_{meta_line}_"


async def send_dataset_selection_prompt(dataset_opts: list[dict], header: str | None = None):
    actions = []
    desc_text = "These are real datasets that match the current tabular CSV workflow. Click one to download and proceed:\n\n"
    for i, ds in enumerate(dataset_opts[:3]):
        source = ds.get("source", "kaggle").lower()
        card_md = render_dataset_card(ds, source)
        reason = ds.get("reason")
        desc_text += f"- {card_md}\n"
        if reason:
            desc_text += f"  - {reason}\n"
        actions.append(
            cl.Action(
                name=f"select_dataset_{i}",
                label=f"Select {i+1}",
                value=ds.get("ref", "error"),
                payload={"ref": ds.get("ref", "error"), "title": ds.get("title", f"Dataset {i+1}")},
            )
        )

    intro = "## Choose Your Real Dataset\n\n"
    if header:
        intro = f"## {header}\n\n"
    await cl.Message(content=intro + desc_text, actions=actions).send()


# ─────────────────────────────────────────────────────────────────────────────
# Chat Start
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    # 1. Handle User Session for Persistence
    if not cl.user_session.get("user"):
        guest = cl.User(identifier="Guest_User")
        cl.user_session.set("user", guest)
    
    # 2. Sync LangGraph thread_id with Chainlit's persistent thread
    # This allows resuming the state of the Graph even if the page is refreshed
    thread_id = cl.context.session.thread_id or str(uuid.uuid4())
    cl.user_session.set("thread_id", thread_id)
    
    # Register session for external FastAPI access
    session_id = "default"
    try: session_id = cl.context.session.id
    except: pass
    _session_threads[session_id] = thread_id
    _session_execution_state[session_id] = {"started": False, "completed": False}

    groq_ok, groq_message = _check_groq_health()

    # 3. Welcome Message (Sent once per session)
    if not cl.user_session.get("welcome_sent"):
        await cl.Message(
            content=(
                "# OmniML - Autonomous HITL Auto-ML Pipeline\n\n"
                "Powered by **Groq** (`openai/gpt-oss-120b`) + **LangGraph** + **Kaggle**.\n\n"
                "---\n"
                "**Describe any machine learning or deep learning problem** and I will help you build it from scratch.\n\n"
                "*Example: `Predict house prices` or `Classify insurance fraud`*"
            )
        ).send()
        cl.user_session.set("welcome_sent", True)

    if not groq_ok and not cl.user_session.get("groq_warning_sent"):
        await cl.Message(
            content=(
                "## Groq Configuration Problem\n\n"
                f"{groq_message}\n\n"
                "OmniML will not start the pipeline until this is fixed."
            )
        ).send()
        cl.user_session.set("groq_warning_sent", True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Message Handler
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_message
async def main(message: cl.Message):
    # ── Update Thread Name for Sidebar Persistence ───────────────────────
    user_query = message.content.strip()
    groq_ok, groq_message = _check_groq_health(force=True)
    if not groq_ok:
        await cl.Message(
            content=(
                "## Pipeline Blocked\n\n"
                f"{groq_message}\n\n"
                "Fix the Groq configuration and restart the app before retrying."
            )
        ).send()
        return

    session_id = cl.user_session.get("id") or "default"
    _session_problems[session_id] = message.content
    _session_execution_state[session_id] = {"started": False, "completed": False}
    cl.user_session.set("problem_statement", user_query)
    if cl.data_layer:
        try:
            await cl.data_layer.update_thread(
                thread_id=cl.context.session.thread_id,
                name=user_query[:30] + ("..." if len(user_query) > 30 else "")
            )
        except Exception:
            pass # Silent fail if data_layer is busy

    thread_id  = cl.user_session.get("thread_id")
    config        = {"configurable": {"thread_id": thread_id}}
    initial_state = {"user_query": user_query}

    await cl.Message(
        content=f"*Launching autonomous pipeline for:* **{user_query}**"
    ).send()

    await send_inline_view(
        session_id,
        view="pipeline_status",
        title="Pipeline Status",
        url="/public/pipeline_status/index.html",
        body="Track the current pipeline stage below while OmniML moves through architecture, data, training, and reporting.",
    )
    
    set_stage('architect', 'active')

    try:
        # ── Phase 1: Run until FIRST HITL interrupt (Model Selection) ────
        async with cl.Step(name=_node_label("architect"), show_input=False) as arch_step:
            arch_step.output = "Analyzing your query and generating architecture candidates..."

            async for chunk in _graph.astream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_state in chunk.items():
                    _update_session_run_state(session_id, node_state)
                    if node_name == "architect":
                        opts = node_state.get("architecture_options", [])
                        graph_payload = node_state.get("graph_architecture_json") or {}
                        count = len(opts)
                        if count == 0 and graph_payload.get("nodes"):
                            count = 1
                        arch_step.output = (
                            f"**{count} Architecture Proposed.**"
                            if count == 1
                            else f"**{count} Architectures Proposed.**"
                        )

        # Phase 1b: HITL - show visual editor
        snapshot     = _graph.get_state(config)
        state_vals   = snapshot.values

        # Pull graph from every possible location, never send empty
        graph = (
            cl.user_session.get("pending_graph") or
            cl.user_session.get("architect_graph") or
            state_vals.get("graph_architecture_json") or
            {
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
                 "data":{"label":"Output","nodeType":"Output",
                         "params":{"units":1,"activation":"sigmoid"}}}
              ],
              "edges": [
                {"id":"e1-2","source":"1","target":"2","animated":True},
                {"id":"e2-3","source":"2","target":"3","animated":True}
              ]
            }
        )
        
        # Validate before injecting
        if not graph.get("nodes"):
            print("[WARNING] Graph has no nodes, using fallback", flush=True)
        
        session_id   = cl.user_session.get("id", "default")
        if hasattr(cl.context, 'session') and hasattr(cl.context.session, 'id'):
            session_id = cl.context.session.id
        
        _session_graphs[session_id] = graph

        await send_inline_view(
            session_id,
            view="architecture_editor",
            title="Human-in-the-Loop: Deep Learning Architect",
            url=f"/public/react_flow_editor/dist/index.html?session_id={session_id}",
            body="I generated a baseline architecture. The Visual Flow Editor is embedded below. Drag in new layers, wire them up, click **Sync Architecture** inside the canvas, then click **Finish Architecture** here.",
            actions=[
                cl.Action(
                    name="finish_architecture",
                    label="Finish Architecture",
                    value="finish",
                    payload={"action": "finish"}
                )
            ]
        )
        return

    except Exception as exc:
        import traceback
        await cl.Message(
            content=f"**Pipeline Error (Phase 1):**\n```\n{traceback.format_exc()}\n```"
        ).send()


# ─────────────────────────────────────────────────────────────────────────────
# Action Callbacks - Architecture Finalization
# ─────────────────────────────────────────────────────────────────────────────
@cl.action_callback("finish_architecture")
async def on_architecture_finished(action: cl.Action):
    thread_id    = cl.user_session.get("thread_id")
    config       = {"configurable": {"thread_id": thread_id}}
    session_id   = cl.context.session.id
    
    # Retrieve graph: prefer synced version (user edited in canvas),
    # fall back to whatever was displayed in the editor (pre-loaded graph)
    payload = _session_graphs.get(session_id) or _session_graphs.get("default", {})
    
    if not payload or len(payload.get("nodes", [])) == 0:
        # Last resort: try to pull from LangGraph snapshot
        snapshot = _graph.get_state(config)
        payload = snapshot.values.get("graph_architecture_json") or {}
    
    if not payload or len(payload.get("nodes", [])) == 0:
        await cl.Message(
            content=(
                "**No architecture found.**\n\n"
                "Please click **SYNC ARCHITECTURE** inside the canvas "
                "first - you should see 'Synced N layers' confirmation "
                "- then click Finish Architecture."
            )
        ).send()
        return

    new_graph = {
        "nodes": payload.get("nodes", []),
        "edges": payload.get("edges", [])
    }
    is_modified = payload.get("isModified", True)
    
    layer_count = len(new_graph['nodes'])
    edge_count  = len(new_graph['edges'])
    
    set_stage('architect', 'complete')
    set_stage('dataset', 'active')
    
    await cl.Message(
        content=(
            f"**Architecture Locked.**\n"
            f"Layers configured: **{layer_count}**\n"
            f"Edges active: **{edge_count}**\n\n"
            f"*Sourcing optimal datasets for your problem...*"
        )
    ).send()

    try:
        # Resume the graph with the confirmed architecture
        _graph.update_state(config, {
            "graph_architecture_json": new_graph,
            "is_architecture_modified": is_modified
        }, as_node="hitl_model_pause")

        dataset_prompt_sent = False
        dataset_stream = _graph.astream(None, config=config, stream_mode="updates")
        try:
            while not dataset_prompt_sent:
                try:
                    chunk = await anext(dataset_stream)
                except StopAsyncIteration:
                    break

                for node_name, node_state in chunk.items():
                    _update_session_run_state(session_id, node_state)
                    if node_name != "dataset_ranker":
                        continue

                    dataset_opts = node_state.get("dataset_options", [])
                    selection_error = node_state.get("dataset_selection_error")
                    if selection_error and not dataset_opts:
                        await cl.Message(
                            content=(
                                "## Dataset Selection Error\n\n"
                                f"{selection_error}"
                            )
                        ).send()
                        dataset_prompt_sent = True
                        break

                    if not dataset_opts:
                        await cl.Message(
                            content=(
                                "## Dataset Selection Error\n\n"
                                "No compatible datasets were found for the current workflow."
                            )
                        ).send()
                        dataset_prompt_sent = True
                        break

                    await send_dataset_selection_prompt(dataset_opts, header="Choose Your Real Dataset")
                    dataset_prompt_sent = True
                    break
        finally:
            await dataset_stream.aclose()

    except Exception:
        import traceback
        await cl.Message(content=f"**Error during sourcing:**\n```\n{traceback.format_exc()}\n```").send()


# ─────────────────────────────────────────────────────────────────────────────
# Action Callbacks - Dataset Selection
# ─────────────────────────────────────────────────────────────────────────────
@cl.action_callback("select_dataset_0")
@cl.action_callback("select_dataset_1")
@cl.action_callback("select_dataset_2")
async def on_dataset_selected(action: cl.Action):
    thread_id    = cl.user_session.get("thread_id")
    config       = {"configurable": {"thread_id": thread_id}}
    payload      = getattr(action, "payload", {}) or {}
    chosen_ref   = payload.get("ref",   action.name)
    chosen_title = payload.get("title", chosen_ref)

    # Resolve session_id for EDA dashboard state
    session_id = "default"
    try:
        if hasattr(cl.context, 'session') and hasattr(cl.context.session, 'id'):
            session_id = cl.context.session.id
    except Exception:
        pass

    await cl.Message(content=f"Selected: **{chosen_title}**\n\n*Downloading dataset...*").send()

    try:
        _graph.update_state(config, {"selected_dataset": chosen_ref}, as_node="hitl_pause")

        async for chunk in _graph.astream(None, config=config, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                _update_session_run_state(session_id, node_state)
                if node_state is None:
                    continue

                if node_name == "dataset_downloader":
                    download_result = node_state.get("dataset_download_result", {})
                    _last_dataset_download_result.clear()
                    _last_dataset_download_result.update(download_result)
                    download_status = download_result.get("status", "download_failed")
                    csv_path = download_result.get("resolved_path", node_state.get("dataset_csv_path", ""))
                    async with cl.Step(name=_node_label("dataset_downloader"), show_input=False) as dl_step:
                        if download_status == "ok":
                            set_stage('dataset', 'complete')
                            set_stage('eda', 'active')
                            dl_step.output = f"Download complete. CSV: `{csv_path}`"
                        else:
                            dl_step.output = f"Download failed. Reason: `{download_result.get('error_message', 'unknown error')}`"

                elif node_name == "dataset_validation":
                    validation = node_state.get("dataset_validation_result", {})
                    async with cl.Step(name="Validate Downloaded Dataset", show_input=False) as validation_step:
                        if validation.get("status") == "ok":
                            validation_step.output = (
                                f"Dataset validated as tabular CSV input. Columns detected: `{validation.get('column_count', 0)}`"
                            )
                        else:
                            validation_step.output = (
                                f"Dataset validation failed. Reason: `{validation.get('message', 'unknown validation error')}`"
                            )

                elif node_name == "eda_analyzer":
                    async with cl.Step(name=_node_label("eda_analyzer"), show_input=False) as eda_step:
                        eda_step.output = "Dataset profiling complete. Generating interactive dashboard and AI narration..."

        # Phase 2.5: HITL - show EDA dashboard
        snapshot     = _graph.get_state(config)
        pending_next = tuple(getattr(snapshot, "next", ()) or ())
        download_result = snapshot.values.get("dataset_download_result", {}) or {}
        validation   = snapshot.values.get("dataset_validation_result", {}) or {}
        acquisition_error = snapshot.values.get("dataset_acquisition_error", {}) or {}
        dataset_opts = snapshot.values.get("dataset_options", []) or []
        eda_data     = snapshot.values.get("eda_data", {})
        narration    = snapshot.values.get("eda_narration", "")
        if download_result:
            _last_dataset_download_result.clear()
            _last_dataset_download_result.update(download_result)
        if validation.get("status") != "ok":
            if download_result.get("status") and download_result.get("status") != "ok":
                title = "Dataset Acquisition Error"
                reason = (
                    acquisition_error.get("message")
                    or download_result.get("error_message")
                    or validation.get("message")
                    or "Dataset acquisition failed."
                )
            else:
                title = "Dataset Validation Error"
                reason = (
                    validation.get("message")
                    or acquisition_error.get("message")
                    or "The dataset was downloaded, but validation did not produce a usable result."
                )
            await cl.Message(
                content=(
                    f"## {title}\n\n"
                    "The selected dataset could not be advanced to the next stage of the current tabular CSV workflow.\n\n"
                    f"Reason: `{reason}`"
                )
            ).send()
            if dataset_opts:
                await send_dataset_selection_prompt(dataset_opts, header="Choose Another Dataset")
            return
        if "hitl_eda_pause" not in pending_next:
            await cl.Message(
                content=(
                    "**EDA Handoff Error**\n\n"
                    "The graph did not pause at the EDA review boundary as expected. "
                    "Please restart the run and retry."
                )
            ).send()
            return
        if not eda_data:
            await cl.Message(
                content=(
                    "**EDA Handoff Error**\n\n"
                    "EDA data was not available after profiling completed, so the dashboard could not be rendered."
                )
            ).send()
            return
        _session_eda_payloads[session_id] = {
            "eda_data": eda_data,
            "eda_narration": narration,
        }
        await send_inline_view(
            session_id,
            view="eda_dashboard",
            title="Live Dataset Analysis",
            url=f"/public/eda_dashboard/index.html?session_id={session_id}",
            body="Review distributions, correlations, outliers, and profiling results in the embedded dashboard below, then click **Proceed to Training Setup** here.",
            actions=[
                cl.Action(name="confirm_eda", label="Proceed to Training Setup", value="confirm", payload={"action": "confirm"})
            ],
        )
        return

    except Exception:
        import traceback
        await cl.Message(content=f"**Error during download/EDA:**\n```\n{traceback.format_exc()}\n```").send()


# ─────────────────────────────────────────────────────────────────────────────
# Action Callbacks - EDA Dashboard Confirmation to Training Config HITL
# ─────────────────────────────────────────────────────────────────────────────
@cl.action_callback("confirm_eda")
async def on_eda_confirmed(action: cl.Action):
    set_stage('eda', 'complete')
    set_stage('config', 'active')
    # Resolve session_id
    session_id = "default"
    try:
        if hasattr(cl.context, 'session') and hasattr(cl.context.session, 'id'):
            session_id = cl.context.session.id
    except Exception:
        pass

    await send_inline_view(
        session_id,
        view="training_config",
        title="Configure Your Training Run",
        url=f"/public/training_config/index.html?session_id={session_id}",
        body="Set your hyperparameters in the embedded training configuration panel below, then click **Validate Settings & Launch Training** here when ready.",
        actions=[
            cl.Action(
                name="launch_training",
                label="Validate Settings & Launch Training",
                value="launch",
                payload={"session_id": session_id}
            )
        ]
    )
    return


# The on_training_launched callback is now simplified to call the helper.
@cl.action_callback("launch_training")
async def on_training_launched(action: cl.Action):
    set_stage('config', 'complete')
    set_stage('hpt', 'active')
    thread_id = cl.user_session.get("thread_id")
    
    session_id = getattr(action, "payload", {}).get("session_id")
    if not session_id:
        try: session_id = cl.context.session.id
        except: session_id = "default"
    _session_execution_state[session_id] = {"started": False, "completed": False}
    
    await cl.Message(content="Initializing training pipeline with your settings...").send()
    await _run_training_pipeline_logic(thread_id, session_id)



# ─────────────────────────────────────────────────────────────────────────────
# Action Callbacks - Execution Mode
# ─────────────────────────────────────────────────────────────────────────────
@cl.action_callback("select_mode_local")
@cl.action_callback("select_mode_cloud")
async def on_execution_mode_selected(action: cl.Action):
    set_stage('engineer', 'complete')
    set_stage('execution', 'active')
    thread_id = cl.user_session.get("thread_id")
    config = {"configurable": {"thread_id": thread_id}}
    payload = getattr(action, "payload", {}) or {}
    mode = payload.get("value", "")
    if not mode:
        mode = "local" if "local" in action.name else "cloud"

    session_id = cl.context.session.id if hasattr(cl.context, "session") and hasattr(cl.context.session, "id") else "default"
    execution_state = _session_execution_state.setdefault(session_id, {"started": False, "completed": False})
    if execution_state.get("completed"):
        await cl.Message(content="Execution already completed for this run. Start a new run to execute again.").send()
        return
    if execution_state.get("started"):
        await cl.Message(content="Execution is already in progress for this run.").send()
        return
    execution_state["started"] = True
    label = "Local Subprocess" if mode == "local" else "Kaggle GPU Deployment"
    await cl.Message(content=f"Starting execution on: **{label}**").send()

    try:
        if mode == "local":
            await send_inline_view(
                session_id,
                view="training_console",
                title="Live Training Progress",
                url="/public/training_console/index.html",
                body="Monitor epochs, logs, and live commentary in the embedded training console below while execution continues.",
            )

        _graph.update_state(config, {"execution_mode": mode}, as_node="execution_choice")

        async for chunk in _graph.astream(None, config=config, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                session_id = cl.context.session.id if hasattr(cl.context, "session") and hasattr(cl.context.session, "id") else "default"
                _update_session_run_state(session_id, node_state)
                if node_state is None:
                    continue

                if node_name == "execution_sandbox":
                    _session_execution_state.setdefault(session_id, {"started": True, "completed": False})["completed"] = True
                    set_stage('execution', 'complete')
                    set_stage('arxiv', 'active')
                    logs = node_state.get("training_logs", "")
                    url = node_state.get("kernel_url", "")

                    async with cl.Step(name=_node_label("execution_sandbox"), show_input=False) as sb_step:
                        if url:
                            sb_step.output = (
                                "**Cloud Execution Active**\n"
                                "Keep this tab open while we poll Kaggle for results.\n\n"
                                f"**Kaggle Link:** [View on Kaggle]({url})\n\n"
                                "---\n"
                                "**Logs retrieved from Cloud:**\n"
                                f"```text\n{logs[:3000]}\n```"
                            )
                        else:
                            sb_step.output = (
                                "**Local Training Complete.**\n"
                                "Final Logs:\n"
                                f"```text\n{logs[:4000]}\n```"
                            )

                elif node_name == "debugger":
                    retries = node_state.get("retry_count", 1)
                    source = node_state.get("active_code_source", "debugger")
                    revision = node_state.get("code_revision", 0)
                    validation = node_state.get("code_validation_result", {}) or {}
                    async with cl.Step(name=_node_label("debugger"), show_input=False) as dbg_step:
                        dbg_step.output = (
                            f"**Error Detected**\n"
                            f"Auto-healing script (Attempt {retries}/3) ...\n"
                            f"Source=`{source}` Revision=`{revision}`\n"
                            f"Validation: {validation.get('message', 'pending')}"
                        )
                        await cl.Message(
                            content=(
                                f"**Execution Failed**. OmniML Debugger Agent has been invoked "
                                f"(Attempt {retries}/3) to regenerate a deterministic training script.\n\n"
                                f"**Source:** `{source}`  \n"
                                f"**Revision:** `{revision}`  \n"
                                f"**Validation:** {validation.get('message', 'pending')}"
                            )
                        ).send()

                elif node_name == "arxiv_comparator":
                    set_stage('arxiv', 'complete')
                    set_stage('deployment', 'active')
                    benchmarks = node_state.get("arxiv_benchmarks", "")
                    benchmark_artifacts = node_state.get("benchmark_artifacts", {})
                    async with cl.Step(name="ArXiv Comparator", show_input=False) as ax_step:
                        ax_step.output = f"**Literature Benchmarks and Gap Analysis Complete:**\n\n{benchmarks}"
                    await cl.Message(
                        content=(
                            "## Benchmark Comparison\n\n"
                            f"{benchmarks}\n\n"
                            f"**Source Status:** `{benchmark_artifacts.get('source_status', 'unknown')}`\n"
                            f"**Cache Hit:** `{benchmark_artifacts.get('cache_hit', False)}`\n"
                            f"**Comparability:** `{benchmark_artifacts.get('comparability', {}).get('score', 'n/a')}`"
                        )
                    ).send()

                elif node_name == "model_deployer":
                    set_stage('deployment', 'complete')
                    set_stage('report', 'active')
                    artifacts = node_state.get("deployment_artifacts")
                    if artifacts:
                        await send_inline_view(
                            session_id,
                            view="deployment_dashboard",
                            title="Model Deployment and Export",
                            url=f"/public/deployment_dashboard/index.html?session_id={session_id}",
                            body="Review exports, benchmark status, fairness artifacts, and compliance reports in the embedded deployment dashboard below.",
                        )
                    else:
                        await cl.Message(content="Model export skipped - no artifacts generated.").send()

                elif node_name == "evaluator":
                    set_stage('report', 'complete')
                    report = node_state.get("final_report", "")

                    elements = []
                    import os
                    if os.path.exists("telemetry_distribution.png"):
                        elements.append(cl.Image(name="Feature Correlation", path="telemetry_distribution.png", display="inline"))
                    if os.path.exists("loss_curve.png"):
                        elements.append(cl.Image(name="Loss Curve", path="loss_curve.png", display="inline"))
                    if os.path.exists("confusion_matrix.png"):
                        elements.append(cl.Image(name="Confusion Matrix", path="confusion_matrix.png", display="inline"))

                    await cl.Message(content="---\n# Final Evaluation Report\n\n" + report, elements=elements).send()

                elif node_name == "fairness_auditor":
                    fairness = node_state.get("fairness_artifacts", {})
                    findings = fairness.get("findings", [])
                    await cl.Message(
                        content=(
                            "## Bias and Fairness Audit\n\n"
                            f"**Status:** `{fairness.get('status', 'unknown')}`\n"
                            f"**Protected Attributes:** `{', '.join(fairness.get('confirmed_sensitive_features', [])) or 'auto-detected / none'}`\n"
                            f"**Findings:** `{len(findings)}`\n\n"
                            f"{fairness.get('narrative', 'No fairness narrative available.')}"
                        )
                    ).send()

                elif node_name == "compliance_renderer":
                    compliance = node_state.get("compliance_artifacts", [])
                    if compliance:
                        lines = ["## Compliance Reports", ""]
                        manifest = _load_manifest_for_session(session_id) or _normalize_manifest_payload(node_state) or {}
                        run_id = manifest.get("run_id") or (_session_runs.get(session_id) or "unknown")
                        for report in compliance:
                            template_id = report.get("template_id", "unknown")
                            completeness = report.get("validation", {}).get("completeness", "unknown")
                            downloads = _compliance_downloads(run_id, template_id)
                            lines.append(
                                f"- `{template_id}` | completeness=`{completeness}` | "
                                f"[md]({downloads['markdown']}) | "
                                f"[html]({downloads['html']}) | "
                                f"[pdf]({downloads['pdf']})"
                            )
                        await cl.Message(content="\n".join(lines)).send()

    except Exception:
        _session_execution_state.setdefault(session_id, {"started": False, "completed": False})["started"] = False
        import traceback
        await cl.Message(content=f"**Error during execution:**\n```\n{traceback.format_exc()}\n```").send()

# Training Pipeline Logic
# ─────────────────────────────────────────────────────────────────────────────

async def _run_training_pipeline_logic(thread_id: str, session_id: str):
    """Refactored logic to resume graph and stream results back to the user."""
    config = {"configurable": {"thread_id": thread_id}}
    terminal_codegen_failure = False

    training_cfg = _merge_training_config(_session_training_configs.get(session_id))
    _session_training_configs[session_id] = training_cfg
    _graph.update_state(config, {"training_config": training_cfg}, as_node="hitl_eda_pause")

    snapshot = _graph.get_state(config)
    pending_next = tuple(getattr(snapshot, "next", ()) or ())
    if "execution_choice" in pending_next:
        arch = snapshot.values.get("architecture", "Unknown Model").split("|")[0].strip()
        await cl.Message(
            content=f"## Compute Strategy\n\nArchitecture: **`{arch}`**\nSelect your environment:",
            actions=[
                cl.Action(name="select_mode_local", label="Local Subprocess", value="local", payload={"value": "local"}),
                cl.Action(name="select_mode_cloud", label="Kaggle Cloud (Free GPU)", value="cloud", payload={"value": "cloud"}),
            ]
        ).send()
        return

    async for chunk in _graph.astream(None, config=config, stream_mode="updates"):
        for node_name, node_state in chunk.items():
            _update_session_run_state(session_id, node_state)
            if node_state is None:
                continue

            if node_name == "hpt_node":
                await send_inline_view(
                    session_id,
                    view="hpt_console",
                    title="Live Hyperparameter Tuning",
                    url="/public/hpt_console/index.html",
                    body="Watch live trial progress, best parameters, and tuning history in the embedded Optuna console below.",
                )

            elif node_name == "engineer":
                set_stage('hpt', 'complete')
                set_stage('engineer', 'active')
                code = node_state.get("generated_code", "")
                preview = "\n".join(code.split("\n")[:20])
                async with cl.Step(name="Engineer", show_input=False) as s:
                    s.output = f"PyTorch and Optuna code generated.\n```python\n{preview}\n# ...\n```"

            elif node_name == "groq_loopfixer":
                code = node_state.get("generated_code", "") or node_state.get("groq_fixed_code", "")
                validation = node_state.get("code_validation_result", {}) or {}
                status = validation.get("status")
                header = (
                    "Code verified and synced with settings."
                    if status == "ok"
                    else f"Repair attempt produced updated code.\nValidation: {validation.get('message', 'pending')}"
                )
                async with cl.Step(name="Loopfixer", show_input=False) as s:
                    s.output = f"{header}\n```python\n{code[:800]}\n# ...\n```"

            elif node_name == "execution_failure":
                failure = node_state.get("training_codegen_error", {}) or {}
                retries = node_state.get("retry_count", 0)
                source = node_state.get("active_code_source", "unknown")
                revision = node_state.get("code_revision", 0)
                set_stage('execution', 'error')
                terminal_codegen_failure = True
                async with cl.Step(name="Training Codegen Failure", show_input=False) as s:
                    s.output = (
                        f"{failure.get('message', 'Training code generation failed after exhausting retries.')}\n\n"
                        f"Retry count: {retries}\n"
                        f"Source: {source}\n"
                        f"Revision: {revision}"
                    )
                await cl.Message(
                    content=(
                        "## Training Code Generation Error\n\n"
                        f"{failure.get('message', 'Training code generation failed after exhausting retries.')}\n\n"
                        f"**Retry count:** `{retries}`  \n"
                        f"**Source:** `{source}`  \n"
                        f"**Revision:** `{revision}`"
                    )
                ).send()

    if terminal_codegen_failure:
        return

    snapshot = _graph.get_state(config)
    arch = snapshot.values.get("architecture", "Unknown Model").split("|")[0].strip()
    await cl.Message(
        content=f"## Compute Strategy\n\nArchitecture: **`{arch}`**\nSelect your environment:",
        actions=[
            cl.Action(name="select_mode_local", label="Local Subprocess", value="local", payload={"value": "local"}),
            cl.Action(name="select_mode_cloud", label="Kaggle Cloud (Free GPU)", value="cloud", payload={"value": "cloud"}),
        ]
    ).send()


custom_paths = [
    "/sync-graph",
    "/hpt-status",
    "/pipeline-status",
    "/training-status",
    "/get-architect-graph",
    "/suggest-architecture",
    "/llm-status",
    "/runtime-diagnostics",
    "/eda-progress",
    "/eda-data",
    "/training-config-store",
    "/training-config-get",
    "/inline-view-state",
    "/launch-pipeline",
    "/deploy-status",
    "/dl-artifact/{filename}",
    "/plugin-catalog",
    "/run-artifacts",
    "/compliance-status",
    "/benchmark-status",
    "/dl-run-artifact/{run_id}/{kind}/{filename}",
]
new_routes = []
catch_all = []
for r in fastapi_app.router.routes:
    if getattr(r, "path", None) in custom_paths:
        new_routes.append(r)
    else:
        catch_all.append(r)
fastapi_app.router.routes = new_routes + catch_all

