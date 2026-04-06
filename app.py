"""
app.py — AnomaLLM v3 Chainlit Frontend
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
import textwrap
import chainlit as cl
from groq import Groq
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from graph import build_graph
from anomallm.persistence import SQLiteDataLayer

from chainlit.server import app as fastapi_app

# ─────────────────────────────────────────────────────────────────────────────
# Global Cache for Iframe Sync
# ─────────────────────────────────────────────────────────────────────────────
_session_graphs:          dict = {}
_session_problems:        dict = {}
_session_training_configs: dict = {}   # session_id -> training config dict
_session_threads:          dict = {}   # session_id -> thread_id string

_pipeline_stages: dict = {}

def set_stage(stage_id: str, status: str):
    _pipeline_stages[stage_id] = status

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
async def deployment_status():
    """Return export artifact status for the deployment dashboard."""
    import os
    export_dir = os.path.join(os.getcwd(), "exports")
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

    return JSONResponse({"ready": True, "meta": meta, "files": files})

@fastapi_app.get("/dl-artifact/{filename}")
async def download_artifact(filename: str):
    """Serve exported model artifacts for download."""
    import os
    from fastapi.responses import FileResponse
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
    """SSE endpoint — streams EDA step events to the live progress dashboard."""
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
    _session_training_configs[session_id] = body
    print(f"[training-config] Stored for session={session_id}: {body}")
    return JSONResponse({"ok": True, "session_id": session_id, "config": body})

@fastapi_app.get("/training-config-get")
async def get_training_config(session_id: str = "default"):
    cfg = _session_training_configs.get(session_id)
    if not cfg:
        return JSONResponse({"ok": False, "error": "No config stored yet"}, status_code=404)
    return JSONResponse({"ok": True, "config": cfg})

@fastapi_app.get("/suggest-architecture")
async def suggest_architecture(session_id: str = "default"):
    try:
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

    # 3. Welcome Message (Sent once per session)
    if not cl.user_session.get("welcome_sent"):
        await cl.Message(
            content=(
                "# 🤖 OmniML — Autonomous HITL Auto-ML Pipeline\n\n"
                "Powered by **Groq** (`openai/gpt-oss-120b`) + **LangGraph** + **Kaggle**.\n\n"
                "---\n"
                "**Describe any machine learning or deep learning problem** and I will help you build it from scratch.\n\n"
                "*Example: `Predict house prices` or `Classify insurance fraud`*"
            )
        ).send()
        cl.user_session.set("welcome_sent", True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Message Handler
# ─────────────────────────────────────────────────────────────────────────────
@cl.on_message
async def main(message: cl.Message):
    # ── Update Thread Name for Sidebar Persistence ───────────────────────
    user_query = message.content.strip()
    session_id = cl.user_session.get("id") or "default"
    _session_problems[session_id] = message.content
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
        content=f"*⚡ Launching autonomous pipeline for:* **{user_query}**"
    ).send()

    pipeline_html = """
    <div style="
      position:fixed; left:16px; top:80px;
      width:160px; z-index:50;
      background:rgba(13,17,23,0.9);
      border:1px solid #21262d;
      border-radius:12px;
      backdrop-filter:blur(16px);
      overflow:hidden;
    ">
      <iframe
        src="/public/pipeline_status/index.html"
        width="160" height="380"
        style="border:none; display:block;">
      </iframe>
    </div>
    """
    await cl.Message(content=pipeline_html).send()
    
    set_stage('architect', 'active')

    try:
        # ── Phase 1: Run until FIRST HITL interrupt (Model Selection) ────
        async with cl.Step(name=_node_label("architect"), show_input=False) as arch_step:
            arch_step.output = "Analyzing your query and generating architecture candidates…"

            async for chunk in _graph.astream(initial_state, config=config, stream_mode="updates"):
                for node_name, node_state in chunk.items():
                    if node_name == "architect":
                        opts = node_state.get("architecture_options", [])
                        arch_step.output = f"**{len(opts)} Architectures Proposed.**"

        # ── Phase 1b: HITL — show Visual Editor ─────────────────
        snapshot     = _graph.get_state(config)
        state_vals   = snapshot.values

        import json, html as html_lib

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
        
        graph_json   = json.dumps(graph)
        escaped_json = html_lib.escape(graph_json, quote=True)
        session_id   = cl.user_session.get("id", "default")
        if hasattr(cl.context, 'session') and hasattr(cl.context.session, 'id'):
            session_id = cl.context.session.id
        
        _session_graphs[session_id] = graph
        
        iframe_html = f"""
        <div style="
          position:relative;
          width:calc(100vw - 80px);
          max-width:1400px;
          height:880px;
          margin-left:calc(-1 * (min(100vw - 80px, 1400px) - 100%) / 2);
          border-radius:20px;
          overflow:hidden;
          border:1px solid #21262d;
          box-shadow:
            0 0 0 1px rgba(99,102,241,0.2),
            0 24px 64px rgba(0,0,0,0.6),
            0 0 120px rgba(99,102,241,0.08);
          margin-top:16px;
          margin-bottom:12px;
        ">
          <div id="graph-data"
               data-payload="{escaped_json}"
               style="display:none;">
          </div>
          <iframe
            id="editor-iframe"
            src="/public/react_flow_editor/dist/index.html?session_id={session_id}"
            width="100%"
            height="100%"
            style="border:none; display:block; margin:0; padding:0;">
          </iframe>
        </div>
        """
        iframe_html = textwrap.dedent(iframe_html).strip()

        await cl.Message(
            content=f"## 🏗️ Human-in-the-Loop: Deep Learning Architect\n\nI have generated a baseline architecture. Use the Visual Flow Editor below to modify it, drag in new layers, and wire them up. Click **Sync Architecture** inside the canvas when you're done.\n\n{iframe_html}",
            actions=[
                cl.Action(
                    name="finish_architecture",
                    label="✅ Finish Architecture",
                    value="finish",
                    payload={"action": "finish"}
                )
            ]
        ).send()

    except Exception as exc:
        import traceback
        await cl.Message(
            content=f"**Pipeline Error (Phase 1):**\n```\n{traceback.format_exc()}\n```"
        ).send()


# ─────────────────────────────────────────────────────────────────────────────
# Action Callbacks — Architecture Finalization
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
                "⚠️ **No architecture found.**\n\n"
                "Please click **SYNC ARCHITECTURE** inside the canvas "
                "first — you should see '✓ Synced N layers' confirmation "
                "— then click Finish Architecture."
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
            f"✅ **Architecture Locked.**\n"
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

        async for chunk in _graph.astream(None, config=config, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                if node_name == "dataset_ranker":
                    dataset_opts = node_state.get("dataset_options", [])
                    architecture = node_state.get("architecture", "")

                    if not dataset_opts:
                        await cl.Message(content="❌ No datasets found. Please try a different query.").send()
                        return

                    actions = []
                    desc_text = "These are **real datasets** from Kaggle — click one to download & proceed:\n\n"
                    for i, ds in enumerate(dataset_opts):
                        source = ds.get("source", "kaggle").lower()
                        card_md = render_dataset_card(ds, source)
                        desc_text += f"- {card_md}\n"
                            
                        actions.append(
                            cl.Action(
                                name=f"select_dataset_{i}",
                                label=f"Select {i+1}",
                                value=ds.get("ref", "error"),
                                payload={"ref": ds.get("ref", "error"), "title": ds.get("title", f"Dataset {i+1}")},
                            )
                        )

                    await cl.Message(
                        content=(
                            "## 🗂️ Human-in-the-Loop: Choose Your Real Dataset\n\n"
                            f"{desc_text}"
                        ),
                        actions=actions,
                    ).send()

    except Exception:
        import traceback
        await cl.Message(content=f"**Error during sourcing:**\n```\n{traceback.format_exc()}\n```").send()


# ─────────────────────────────────────────────────────────────────────────────
# Action Callbacks — Dataset Selection
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

    # Resolve session_id for EDA iframe
    session_id = "default"
    try:
        if hasattr(cl.context, 'session') and hasattr(cl.context.session, 'id'):
            session_id = cl.context.session.id
    except Exception:
        pass

    await cl.Message(content=f"✅ Selected: **{chosen_title}**\n\n*Downloading dataset...*").send()

    try:
        _graph.update_state(config, {"selected_dataset": chosen_ref}, as_node="hitl_pause")

        async for chunk in _graph.astream(None, config=config, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                if node_state is None:
                    continue

                if node_name == "dataset_downloader":
                    set_stage('dataset', 'complete')
                    set_stage('eda', 'active')
                    csv_path = node_state.get("dataset_csv_path", "")
                    async with cl.Step(name=_node_label("dataset_downloader"), show_input=False) as dl_step:
                        dl_step.output = f"✅ Download complete! CSV: `{csv_path}`"

                elif node_name == "eda_analyzer":
                    async with cl.Step(name=_node_label("eda_analyzer"), show_input=False) as eda_step:
                        eda_step.output = "🔍 Dataset profiling complete. Generating interactive dashboard and AI narration..."

        # ── Phase 2.5: HITL — show EDA Dashboard ─────────────────
        snapshot     = _graph.get_state(config)
        eda_data     = snapshot.values.get("eda_data", {})
        narration    = snapshot.values.get("eda_narration", "")

        import json
        import textwrap
        import base64
        
        # Base64 encode to prevent XSS and character breakage in HTML injection
        eda_json_str  = json.dumps(eda_data)
        eda_b64       = base64.b64encode(eda_json_str.encode()).decode()
        narration_b64 = base64.b64encode(narration.encode()).decode()

        # Build EDA Iframe — full-width breakout + passes session_id for SSE streaming
        eda_iframe_html = textwrap.dedent(f"""\
<div style="margin-left:calc(-50vw + 50%); margin-right:calc(-50vw + 50%); width:100vw; height:1100px; border-radius:0; overflow:hidden; border-top:1px solid #21262d; border-bottom:1px solid #21262d; position:relative; margin-top:20px; margin-bottom:20px; box-shadow:0 24px 80px rgba(0,0,0,0.6);">
    <div id="eda-data-b64" style="display:none;">{eda_b64}</div>
    <div id="eda-narration-b64" style="display:none;">{narration_b64}</div>
    <iframe id="eda-iframe" src="/public/eda_dashboard/index.html?session_id={session_id}" width="100%" height="100%" style="border:none; margin:0; padding:0; display:block;"></iframe>
</div>""").strip()

        await cl.Message(
            content=(
                f"## 📊 OmniML — Live Dataset Analysis\n\n"
                f"Watch the AI profile your dataset in real time — distributions, correlations, outliers, and insights.\n\n"
                f"{eda_iframe_html}"
            ),
            actions=[
                cl.Action(name="confirm_eda", label="🚀 Proceed to Training Setup", value="confirm", payload={"action": "confirm"})
            ]
        ).send()

    except Exception:
        import traceback
        await cl.Message(content=f"**Error during download/EDA:**\n```\n{traceback.format_exc()}\n```").send()


# ─────────────────────────────────────────────────────────────────────────────
# Action Callbacks — EDA Dashboard Confirmation → Training Config HITL
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

    import textwrap
    config_iframe = textwrap.dedent(f"""\
<div style="margin-left:calc(-50vw + 50%); margin-right:calc(-50vw + 50%); width:100vw; height:820px; border-top:1px solid #21262d; border-bottom:1px solid #21262d; position:relative; margin-top:20px; margin-bottom:20px; box-shadow:0 24px 80px rgba(0,0,0,0.6);">
    <iframe src="/public/training_config/index.html?session_id={session_id}" width="100%" height="100%" style="border:none; display:block;"></iframe>
</div>""").strip()

    await cl.Message(
        content=(
            "## ⚙️ Configure Your Training Run\n\n"
            "Set your hyperparameters below — epochs, test split, optimizer, regularization, and more. "
            "When you're ready, click **Launch Training**.\n\n"
            f"{config_iframe}"
        ),
        actions=[
            cl.Action(
                name="launch_training",
                label="🚀 Validate Settings & Launch Training",
                value="launch",
                payload={"session_id": session_id}
            )
        ]
    ).send()


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
    
    await cl.Message(content="🚀 Initializing training pipeline with your settings...").send()
    await _run_training_pipeline_logic(thread_id, session_id)



# ─────────────────────────────────────────────────────────────────────────────
# Action Callbacks — Execution Mode
# ─────────────────────────────────────────────────────────────────────────────
@cl.action_callback("select_mode_local")
@cl.action_callback("select_mode_cloud")
async def on_execution_mode_selected(action: cl.Action):
    set_stage('engineer', 'complete')
    set_stage('execution', 'active')
    thread_id = cl.user_session.get("thread_id")
    config    = {"configurable": {"thread_id": thread_id}}
    payload = getattr(action, "payload", {}) or {}
    mode    = payload.get("value", "")
    if not mode:
        mode = "local" if "local" in action.name else "cloud"

    label = "💻 Local Subprocess" if mode == "local" else "☁️ Kaggle GPU Deployment"
    await cl.Message(content=f"🚀 Starting execution on: **{label}**").send()

    try:
        if mode == "local":
            iframe_html_train = """<div style="width:100%; height:800px; border-radius:12px; overflow:hidden; border:2px solid #30363d; margin-top:20px;">
                        <iframe src="/public/training_console/index.html" width="100%" height="100%" style="border:none;"></iframe>
                    </div>"""
            await cl.Message(
                content=f"## 📈 Live Training Progress\n\n{iframe_html_train}"
            ).send()

        _graph.update_state(config, {"execution_mode": mode}, as_node="execution_choice")

        # ── Final leg: Execution → Evaluator ───────────────────────────────
        async for chunk in _graph.astream(None, config=config, stream_mode="updates"):
            for node_name, node_state in chunk.items():
                # LangGraph emits None for interrupt/pause nodes — skip cleanly
                if node_state is None:
                    continue

                if node_name == "execution_sandbox":
                    set_stage('execution', 'complete')
                    set_stage('arxiv', 'active')
                    logs = node_state.get("training_logs", "")
                    url  = node_state.get("kernel_url", "")

                    async with cl.Step(name=_node_label("execution_sandbox"), show_input=False) as sb_step:
                        if url:
                            sb_step.output = f"☁️ **Cloud Execution Active**\nKeep this tab open while we poll Kaggle for results.\n\n**Kaggle Link:** [View on Kaggle]({url})\n\n---\n**Logs retrieved from Cloud:**\n```text\n{logs[:3000]}\n```"
                        else:
                            sb_step.output = f"✅ **Local Training Complete.**\nFinal Logs:\n```text\n{logs[:4000]}\n```"

                elif node_name == "debugger":
                    retries = node_state.get("retry_count", 1)
                    async with cl.Step(name=_node_label("debugger"), show_input=False) as dbg_step:
                        dbg_step.output = f"⚠️ **Error Detected**\nAuto-healing script (Attempt {retries}/3) ...\nRewriting PyTorch logic based on Traceback."
                        await cl.Message(content=f"⚠️ **Execution Failed**. OmniML Debugger Agent has been invoked (Attempt {retries}/3) to automatically rewrite and heal the script.").send()

                elif node_name == "arxiv_comparator":
                    set_stage('arxiv', 'complete')
                    set_stage('deployment', 'active')
                    benchmarks = node_state.get("arxiv_benchmarks", "")
                    async with cl.Step(name="📚 ArXiv Comparator", show_input=False) as ax_step:
                        ax_step.output = f"**Literature Benchmarks & Gap Analysis Complete:**\n\n{benchmarks}"
                    await cl.Message(content=f"## 📚 ArXiv Publication Comparison\n\n{benchmarks}").send()

                elif node_name == "model_deployer":
                    set_stage('deployment', 'complete')
                    set_stage('report', 'active')
                    artifacts = node_state.get("deployment_artifacts")
                    if artifacts:
                        iframe_html = '<iframe src="/public/deployment_dashboard/index.html" style="width:100%;height:680px;border:none;border-radius:12px;"></iframe>'
                        await cl.Message(content=f"## 🚀 Model Deployment & Export\n\n{iframe_html}").send()
                    else:
                        await cl.Message(content="⚠️ Model export skipped — no artifacts generated.").send()

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

                    # NOTE: cl.File is explicitly banned as it triggers a t.startsWith generic file-handler crash in the React UI!
                    await cl.Message(content="---\n# 📋 Final Evaluation Report\n\n" + report, elements=elements).send()



    except Exception:
        import traceback
        await cl.Message(content=f"**Error during execution:**\n```\n{traceback.format_exc()}\n```").send()

# ─────────────────────────────────────────────────────────────────────────────
# Training Pipeline Logic
# ─────────────────────────────────────────────────────────────────────────────

async def _run_training_pipeline_logic(thread_id: str, session_id: str):
    """Refactored logic to resume graph and stream results back to the user."""
    # Since this might run outside a direct callback, we ensure we have 
    # the correct LangGraph config.
    config = {"configurable": {"thread_id": thread_id}}
    
    # 1. Fetch training config
    defaults = {
        "epochs": 50, "test_size": 0.2, "batch_size": 64,
        "optimizer": "adam", "lr": 0.001, "early_stop": True,
        "dropout": True, "class_weights": True, "shuffle": True,
        "hpt_trials": 15, "seed": 42
    }
    training_cfg = _session_training_configs.get(session_id, defaults)
    training_cfg = {**defaults, **training_cfg}

    # 2. Update state
    _graph.update_state(config, {"training_config": training_cfg}, as_node="hitl_eda_pause")
    
    # 3. Stream graph execution
    async for chunk in _graph.astream(None, config=config, stream_mode="updates"):
        for node_name, node_state in chunk.items():
            if node_state is None: continue
            
            # NOTE: We can only send messages if cl.context is active. 
            # In a background task, we might need cl.context_var.set().
            # For brevity, we assume the user is still in the chat.
            
            if node_name == "hpt_node":
                html = """<div style="width:100%; height:800px; border-radius:12px; overflow:hidden; border:2px solid #30363d; margin-top:20px;">
                            <iframe src="/public/hpt_console/index.html" width="100%" height="100%" style="border:none;"></iframe>
                        </div>"""
                await cl.Message(content=f"## 🎛️ Live Hyperparameter Tuning\n\n{html}").send()

            elif node_name == "engineer":
                set_stage('hpt', 'complete')
                set_stage('engineer', 'active')
                code = node_state.get("generated_code", "")
                preview = "\n".join(code.split("\n")[:20])
                async with cl.Step(name="Engineer", show_input=False) as s:
                    s.output = f"✅ PyTorch + Optuna Code generated.\n```python\n{preview}\n# ...\n```"

            elif node_name == "groq_loopfixer":
                code = node_state.get("groq_fixed_code", "")
                async with cl.Step(name="Loopfixer", show_input=False) as s:
                    s.output = f"✅ Code verified & synced with settings.\n```python\n{code[:800]}\n# ...\n```"

    # 4. Final choice
    snapshot = _graph.get_state(config)
    arch = snapshot.values.get("architecture", "Unknown Model").split("|")[0].strip()
    await cl.Message(
        content=f"## ☁️ Compute Strategy\n\nArchitecture: **`{arch}`**\nSelect your environment:",
        actions=[
            cl.Action(name="select_mode_local", label="💻 Local Subprocess", value="local", payload={"value": "local"}),
            cl.Action(name="select_mode_cloud", label="☁️ Kaggle Cloud (Free GPU)", value="cloud", payload={"value": "cloud"}),
        ]
    ).send()



custom_paths = ["/sync-graph", "/hpt-status", "/pipeline-status", "/training-status", "/get-architect-graph", "/suggest-architecture", "/eda-progress", "/training-config-store", "/training-config-get", "/launch-pipeline", "/deploy-status", "/dl-artifact/{filename}"]
new_routes = []
catch_all = []
for r in fastapi_app.router.routes:
    if getattr(r, "path", None) in custom_paths:
        new_routes.append(r)
    else:
        catch_all.append(r)
fastapi_app.router.routes = new_routes + catch_all

