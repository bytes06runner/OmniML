"""
tools.py — AnomaLLM v3 Real Kaggle Tool Suite
==============================================
Provides three capabilities:
  1. kaggle_auth_setup()       — Writes ~/.kaggle/kaggle.json at startup
  2. kaggle_search_tool()      — Searches Kaggle for real datasets
  3. kaggle_download_tool()    — Downloads the selected dataset CSV to disk

Author: AnomaLLM v3 / Antigravity
"""

import os
import sys
import re
import json
import builtins
import zipfile
import glob
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.tools import tool

# Resolve absolute path to 'kaggle' based on the active python interpreter

# ─────────────────────────────────────────────
# Load env vars immediately so all tools see them
# ─────────────────────────────────────────────
load_dotenv()

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "")
KAGGLE_KEY      = os.environ.get("KAGGLE_KEY", "")
DATA_DIR        = Path("data")


def _safe_print(message: str) -> None:
    try:
        builtins.print(message)
    except UnicodeEncodeError:
        builtins.print(message.encode("ascii", errors="ignore").decode("ascii"))


print = _safe_print


def _resolve_kaggle_cli_path() -> str:
    resolved = shutil.which("kaggle")
    if resolved:
        return resolved

    py_dir = Path(sys.executable).resolve().parent
    if os.name == "nt":
        candidates = [
            py_dir / "Scripts" / "kaggle.exe",
            py_dir / "kaggle.exe",
        ]
    else:
        candidates = [py_dir / "kaggle"]

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(candidates[0])


def _kaggle_env() -> dict:
    return {**os.environ, "KAGGLE_USERNAME": KAGGLE_USERNAME, "KAGGLE_KEY": KAGGLE_KEY}


def _classify_download_error(message: str, source: str = "kaggle") -> str:
    text = (message or "").lower()
    if any(token in text for token in ["proxyerror", "proxy", "connectionerror", "timed out", "timeout", "10061", "temporarily unavailable"]):
        return "network_failed"
    if any(token in text for token in ["401", "403", "unauthorized", "forbidden", "invalid api key", "credentials", "authenticate"]):
        return "auth_failed"
    if "no csv files found" in text or "unsupported format" in text:
        return "unsupported_format"
    if source == "huggingface" and any(token in text for token in ["builderconfig", "doesn't exist on the hub", "contains no splits"]):
        return "unsupported_format"
    return "download_failed"


def _download_result(
    *,
    status: str,
    source: str,
    dataset_ref: str,
    resolved_path: str = "",
    detected_format: str = "",
    error_message: str = "",
) -> dict:
    return {
        "status": status,
        "source": source,
        "dataset_ref": dataset_ref,
        "resolved_path": resolved_path,
        "detected_format": detected_format,
        "error_message": error_message,
    }


def _detect_tabular_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        return suffix.lstrip(".")
    return suffix.lstrip(".") or "unknown"


def _kaggle_preflight() -> tuple[bool, str, str]:
    kaggle_cli = _resolve_kaggle_cli_path()
    if not Path(kaggle_cli).exists():
        return False, kaggle_cli, f"Kaggle CLI not found at {kaggle_cli}"
    if not KAGGLE_USERNAME or not KAGGLE_KEY:
        return False, kaggle_cli, "KAGGLE_USERNAME or KAGGLE_KEY is not configured"
    return True, kaggle_cli, ""


KAGGLE_CLI_PATH = _resolve_kaggle_cli_path()


# ─────────────────────────────────────────────
# 1.  Auth Setup — called once at startup
# ─────────────────────────────────────────────
def kaggle_auth_setup() -> bool:
    """
    Write ~/.kaggle/kaggle.json from env vars so the kaggle library
    authenticates correctly regardless of how the process was launched.

    Returns True on success, False on failure.
    """
    try:
        if os.environ.get("OMNIML_SKIP_KAGGLE_AUTH") == "1" or "unittest" in sys.modules:
            return False

        kaggle_dir  = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"

        if not KAGGLE_USERNAME or not KAGGLE_KEY:
            _safe_print("[tools] WARNING: KAGGLE_USERNAME or KAGGLE_KEY not set in .env")
            return False

        kaggle_dir.mkdir(parents=True, exist_ok=True)
        kaggle_json.write_text(
            json.dumps({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}),
            encoding="utf-8",
        )
        # Kaggle API requires 600 permissions on this file
        kaggle_json.chmod(0o600)

        # Also ensure env vars are set for subprocesses
        os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
        os.environ["KAGGLE_KEY"]      = KAGGLE_KEY

        print(f"[tools] ✅ Kaggle credentials written to {kaggle_json}")
        return True

    except Exception as exc:
        print(f"[tools] ❌ kaggle_auth_setup failed: {exc}")
        return False


# ─────────────────────────────────────────────
# 2.  Kaggle Search Tool
# ─────────────────────────────────────────────
@tool
def kaggle_search_tool(query: str) -> str:
    """
    Search Kaggle for datasets matching the given query using the Kaggle Python API.
    Returns JSON array of rich dataset dicts.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        import json
        
        api = KaggleApi()
        api.authenticate()
        
        clean_query = query.replace(",", " ").replace(";", " ").replace(".", " ")
        clean_query = " ".join(clean_query.split()[:3])
        print(f"[kaggle_search_tool] Searching Kaggle via API for: '{clean_query}'")

        datasets = api.dataset_list(search=clean_query, sort_by="votes")
        
        if not datasets:
            return json.dumps([{"error": f"No Kaggle datasets found for {clean_query}"}])
            
        results = []
        for ds in datasets[:3]:
            try:
                # KaggleApi Dataset object attributes
                ref = getattr(ds, "ref", None) or f"{ds.ownerRef}/{ds.ref}"
                title = getattr(ds, "title", ref)
                url = f"https://www.kaggle.com/datasets/{ref}"
                
                size_mb = 0
                if getattr(ds, "totalBytes", None):
                    size_mb = round(ds.totalBytes / 1.0e6, 1)
                elif getattr(ds, "size", None):
                    size_bytes = ds.size
                    if type(size_bytes) is str and getattr(ds, "totalBytes", None):
                        pass # Size is a string e.g. '10MB'
                
                results.append({
                    "source": "kaggle",
                    "title": title,
                    "ref": ref,
                    "url": url,
                    "downloads": getattr(ds, "downloadCount", 0),
                    "votes": getattr(ds, "voteCount", 0),
                    "size_mb": size_mb,
                    "last_updated": str(getattr(ds, "lastUpdated", ""))[:10],
                    "license": getattr(ds, "licenseName", "Unknown"),
                    # Note: rows and cols might not be on the summary list dataset object.
                })
            except Exception as e:
                print(f"Dataset parsing skipped: {e}")
                
        if not results:
            return json.dumps([{"error": "Failed to parse API dataset returns."}])
            
        return json.dumps(results)

    except Exception as exc:
        return json.dumps([{"error": f"Kaggle search failed: {exc}"}])


# ─────────────────────────────────────────────
# 3.  Kaggle Dataset Downloader Tool
# ─────────────────────────────────────────────
@tool
def _legacy_kaggle_download_tool_string(dataset_ref: str) -> dict:
    """
    Download a Kaggle dataset by its ref (e.g. 'owner/dataset-slug'),
    unzip it, and return the absolute path to the first CSV file found.

    Uses the kaggle CLI subprocess so it works with large datasets
    that the Python API times out on.

    Returns:
        Absolute path string to the CSV  —  or an "ERROR: ..." string on failure.
    """
    try:
        # Sanitise ref to use as a directory name
        slug     = dataset_ref.replace("/", "__")
        dest_dir = DATA_DIR / slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        # ── Try kaggle CLI first (most reliable for large files) ────────────
        print(f"[tools] ⬇️  Downloading dataset: {dataset_ref} → {dest_dir}")
        result = subprocess.run(
            [
                KAGGLE_CLI_PATH, "datasets", "download",
                "-d", dataset_ref,
                "-p", str(dest_dir),
                "--unzip",
            ],
            capture_output=True,
            text=True,
            timeout=300,   # 5 minute download limit
            env={**os.environ, "KAGGLE_USERNAME": KAGGLE_USERNAME, "KAGGLE_KEY": KAGGLE_KEY},
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "kaggle CLI returned non-zero exit code")

        # ── Find the first CSV in the destination ───────────────────────────
        csv_files = sorted(glob.glob(str(dest_dir / "**" / "*.csv"), recursive=True))

        if not csv_files:
            # Maybe it's a zip that needs manual extraction
            zip_files = list(dest_dir.glob("*.zip"))
            for zf in zip_files:
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(dest_dir)
            csv_files = sorted(glob.glob(str(dest_dir / "**" / "*.csv"), recursive=True))

        if not csv_files:
            return f"ERROR: Dataset {dataset_ref} downloaded but no CSV files found in {dest_dir}"

        # Return the largest CSV (most likely the main data file)
        csv_files.sort(key=lambda p: os.path.getsize(p), reverse=True)
        chosen = csv_files[0]
        size_mb = os.path.getsize(chosen) / (1024 * 1024)
        print(f"[tools] ✅ CSV resolved: {chosen}  ({size_mb:.1f} MB)")
        return str(Path(chosen).absolute())

    except subprocess.TimeoutExpired:
        return f"ERROR: Download timed out for dataset {dataset_ref} (>5 min)"

    except Exception as exc:
        # ── Fallback: use the NASA C-MAPSS dataset already in workspace ─────
        fallback_path = Path("__disabled_fallback__")
        if fallback_path.exists():
            print(f"[tools] ⚠️  Kaggle download failed ({exc}). Using NASA C-MAPSS fallback.")
            return str(fallback_path.absolute())
        return f"ERROR: Download failed for {dataset_ref} — {type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────
# 4.  Kaggle Kernel Push Tool (CLOUD EXECUTION)
# ─────────────────────────────────────────────
@tool
def kaggle_download_tool(dataset_ref: str) -> dict:
    """
    Download a Kaggle dataset by its ref and return a structured result.
    """
    ok, kaggle_cli, preflight_error = _kaggle_preflight()
    if not ok:
        return _download_result(
            status=_classify_download_error(preflight_error),
            source="kaggle",
            dataset_ref=dataset_ref,
            error_message=preflight_error,
        )

    try:
        slug = dataset_ref.replace("/", "__")
        dest_dir = DATA_DIR / slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        print(f"[tools] Downloading dataset: {dataset_ref} -> {dest_dir}")
        result = subprocess.run(
            [
                kaggle_cli, "datasets", "download",
                "-d", dataset_ref,
                "-p", str(dest_dir),
                "--unzip",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            env=_kaggle_env(),
        )

        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "kaggle CLI returned non-zero exit code")

        csv_files = sorted(glob.glob(str(dest_dir / "**" / "*.csv"), recursive=True))
        if not csv_files:
            zip_files = list(dest_dir.glob("*.zip"))
            for zf in zip_files:
                with zipfile.ZipFile(zf, "r") as archive:
                    archive.extractall(dest_dir)
            csv_files = sorted(glob.glob(str(dest_dir / "**" / "*.csv"), recursive=True))

        if not csv_files:
            return _download_result(
                status="unsupported_format",
                source="kaggle",
                dataset_ref=dataset_ref,
                error_message=f"Dataset {dataset_ref} downloaded but no CSV files were found in {dest_dir}",
            )

        csv_files.sort(key=lambda p: os.path.getsize(p), reverse=True)
        chosen = Path(csv_files[0]).absolute()
        size_mb = os.path.getsize(chosen) / (1024 * 1024)
        print(f"[tools] CSV resolved: {chosen} ({size_mb:.1f} MB)")
        return _download_result(
            status="ok",
            source="kaggle",
            dataset_ref=dataset_ref,
            resolved_path=str(chosen),
            detected_format=_detect_tabular_format(chosen),
        )

    except subprocess.TimeoutExpired:
        return _download_result(
            status="network_failed",
            source="kaggle",
            dataset_ref=dataset_ref,
            error_message=f"Download timed out for dataset {dataset_ref} (>5 min)",
        )
    except Exception as exc:
        error_message = f"Download failed for {dataset_ref} - {type(exc).__name__}: {exc}"
        return _download_result(
            status=_classify_download_error(error_message),
            source="kaggle",
            dataset_ref=dataset_ref,
            error_message=error_message,
        )


@tool
def kaggle_push_tool(code: str, dataset_ref: str, title: str = "AnomaLLM v3 Auto-ML") -> str:
    """
    Deploy the PyTorch code to a Kaggle kernel for cloud execution (GPU enabled).

    Args:
        code: The raw PyTorch Python code.
        dataset_ref: The Kaggle dataset reference (e.g. 'owner/slug').
        title: Descriptive title for the Kaggle kernel.

    Returns:
        The URL of the newly created Kaggle kernel or an "ERROR: ..." string.
    """
    import time
    try:
        # ── Setup temporary kernel directory ────────────────────────────────
        slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
        # Add timestamp to make it unique
        slug = f"{slug}-{int(time.time())}"
        
        kernel_dir = DATA_DIR / "kernels" / slug
        kernel_dir.mkdir(parents=True, exist_ok=True)
        
        # ── Write code script ───────────────────────────────────────────────
        code_file = kernel_dir / "script.py"
        code_file.write_text(code, encoding="utf-8")
        
        # ── Create metadata ─────────────────────────────────────────────────
        # Ensure dataset_ref is in the right format
        metadata = {
            "id": f"{KAGGLE_USERNAME}/{slug}",
            "title": f"{title} ({int(time.time())})",
            "code_file": "script.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": "true",
            "enable_gpu": "true",
            "enable_tpu": "false",
            "enable_internet": "true",
            "dataset_sources": [dataset_ref],
            "competition_sources": [],
            "kernel_sources": [],
            "model_sources": []
        }
        
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        
        # ── Push to Kaggle ──────────────────────────────────────────────────
        print(f"[tools] ☁️  Pushing kernel to Kaggle: {slug}")
        result = subprocess.run(
            [KAGGLE_CLI_PATH, "kernels", "push", "-p", str(kernel_dir)],
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "KAGGLE_USERNAME": KAGGLE_USERNAME, "KAGGLE_KEY": KAGGLE_KEY},
        )
        
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "kaggle CLI (push) returned non-zero exit code")
        
        # ── Construct the URL ───────────────────────────────────────────────
        url = f"https://www.kaggle.com/code/{KAGGLE_USERNAME}/{slug}"
        print(f"[tools] ✅ Kaggle Kernel pushed successfully: {url}")
        
        # Return a JSON-like string with all metadata
        return json.dumps({
            "url": url,
            "ref": f"{KAGGLE_USERNAME}/{slug}",
            "slug": slug,
            "username": KAGGLE_USERNAME
        })

    except Exception as exc:
        return f"ERROR: Kaggle push failed — {type(exc).__name__}: {exc}"


# ─────────────────────────────────────────────
# 5.  Kaggle Status & Output Tools (MONITORING)
# ─────────────────────────────────────────────
@tool
def kaggle_status_tool(kernel_ref: str) -> str:
    """
    Check the current status of a Kaggle kernel.
    Returns: 'queued', 'running', 'complete', or 'error'.
    """
    try:
        result = subprocess.run(
            [KAGGLE_CLI_PATH, "kernels", "status", kernel_ref],
            capture_output=True,
            text=True,
            timeout=15,
            env={**os.environ, "KAGGLE_USERNAME": KAGGLE_USERNAME, "KAGGLE_KEY": KAGGLE_KEY},
        )
        # Status output format: "<kernel_ref> has status '<status>'"
        out = result.stdout.strip()
        if "has status" in out:
            status = out.split("'")[1]
            return status
        return f"UNKNOWN: {out}"
    except Exception as exc:
        return f"ERROR: Status check failed: {exc}"


@tool
def kaggle_output_tool(kernel_ref: str) -> str:
    """
    Download the output logs (stdout/stderr) from a completed Kaggle kernel.
    """
    try:
        slug     = kernel_ref.split("/")[-1]
        temp_dir = DATA_DIR / "kernels" / f"output_{slug}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Download output
        subprocess.run(
            [KAGGLE_CLI_PATH, "kernels", "output", kernel_ref, "-p", str(temp_dir)],
            capture_output=True,
            text=True,
            timeout=30,
            env={**os.environ, "KAGGLE_USERNAME": KAGGLE_USERNAME, "KAGGLE_KEY": KAGGLE_KEY},
        )
        
        # Try to find a log file or result file. Kaggle logs stdout to a specific file.
        # Often it's named 'script.log' or similar if it's a script kernel.
        log_files = glob.glob(str(temp_dir / "*.log")) + glob.glob(str(temp_dir / "*.txt"))
        
        if not log_files:
            return f"✅ CLOUD EXECUTION COMPLETE. (No explicit log files returned, check manually: https://www.kaggle.com/code/{kernel_ref})"

        # Sort by latest
        log_files.sort(key=os.path.getmtime, reverse=True)
        content = Path(log_files[0]).read_text(encoding="utf-8")
        return content[:5000] # Cap for LLM context

    except Exception as exc:
        return f"ERROR: Output retrieval failed: {exc}"


# ─────────────────────────────────────────────
# 6.  HuggingFace Search Tool
# ─────────────────────────────────────────────
@tool
def hf_search_tool(query: str) -> str:
    """
    Search HuggingFace Datasets Hub.
    Returns JSON array of dataset metadata.
    """
    try:
        from huggingface_hub import HfApi
        import json
        
        clean_query = query.replace(",", " ").replace(";", " ").replace(".", " ")
        search_query = " ".join(clean_query.split()[:3])
        print(f"[hf_search_tool] Searching HuggingFace for '{search_query}'")
        
        api = HfApi()
        datasets = list(api.list_datasets(
            search=search_query,
            sort="downloads",
            direction=-1,
            limit=5
        ))
        
        if not datasets:
            return json.dumps([{"error": f"No HF datasets found for {search_query}"}])
            
        results = []
        for ds in datasets[:3]:
            try:
                repo_id = ds.id
                info = api.dataset_info(repo_id)
                card = getattr(info, "cardData", {}) or {}
                
                size_cat = card.get("size_categories", ["unknown"])
                size_cat = size_cat[0] if isinstance(size_cat, list) and size_cat else "unknown"
                
                results.append({
                    "source": "huggingface",
                    "title": repo_id.split("/")[-1],
                    "ref": repo_id,
                    "dataset_id": repo_id,
                    "url": f"https://huggingface.co/datasets/{repo_id}",
                    "downloads": getattr(info, "downloads", 0),
                    "likes": getattr(info, "likes", 0),
                    "votes": getattr(info, "likes", 0),
                    "last_updated": str(getattr(info, "lastModified", ""))[:10],
                    "size_category": size_cat,
                    "description": getattr(info, "description", "")[:200]
                })
            except Exception as e:
                print(f"HF Dataset parse skipped: {e}")
                
        if not results:
            return json.dumps([{"error": "Failed to parse HF API dataset returns."}])
        return json.dumps(results)
    
    except Exception as exc:
        return json.dumps([{"error": f"HF search failed: {exc}"}])

# ─────────────────────────────────────────────
# 7.  HuggingFace Download Tool
# ─────────────────────────────────────────────
@tool
def _legacy_hf_download_tool_string(dataset_ref: str) -> str:
    """
    Download a dataset from HuggingFace, save it as a CSV, and return the absolute path.
    """
    try:
        from datasets import load_dataset
        import pandas as pd
        
        slug = dataset_ref.replace("/", "__")
        dest_dir = DATA_DIR / slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        csv_path = dest_dir / "data.csv"
        
        if csv_path.exists():
            return str(csv_path.absolute())
            
        print(f"[hf_download_tool] Downloading HF dataset: {dataset_ref}...")
        try:
            ds = load_dataset(dataset_ref, split="train")
        except Exception as e:
            print(f"[hf_download_tool] Standard split failed: {e}. Trying first available...")
            ds = load_dataset(dataset_ref)
            keys = list(ds.keys())
            if not keys:
                raise ValueError("Dataset contains no splits.")
            ds = ds[keys[0]]
            
        df = ds.to_pandas()
        
        # Enforce <500MB roughly in memory via dataframe bytes
        mem_usage = df.memory_usage(deep=True).sum()
        if mem_usage > 500 * 1024 * 1024:
            return f"ERROR: Dataset {dataset_ref} exceeds 500MB limit ({mem_usage / 1024 / 1024:.1f} MB)."
            
        df.to_csv(csv_path, index=False)
        print(f"[hf_download_tool] ✅ Saved CSV to {csv_path} ({mem_usage / 1024 / 1024:.1f} MB)")
        
        return str(csv_path.absolute())
        
    except Exception as exc:
        return f"ERROR: HuggingFace download failed for {dataset_ref} — {type(exc).__name__}: {exc}"

# ─────────────────────────────────────────────
# 8.  ArXiv Search Tool (Sprint 4)
# ─────────────────────────────────────────────
@tool
def hf_download_tool(dataset_ref: str) -> dict:
    """
    Download a dataset from HuggingFace and return a structured result.
    """
    try:
        from datasets import load_dataset
        import pandas as pd

        slug = dataset_ref.replace("/", "__")
        dest_dir = DATA_DIR / slug
        dest_dir.mkdir(parents=True, exist_ok=True)

        csv_path = dest_dir / "data.csv"
        if csv_path.exists():
            return _download_result(
                status="ok",
                source="huggingface",
                dataset_ref=dataset_ref,
                resolved_path=str(csv_path.absolute()),
                detected_format=_detect_tabular_format(csv_path),
            )

        print(f"[hf_download_tool] Downloading HF dataset: {dataset_ref}...")
        try:
            ds = load_dataset(dataset_ref, split="train")
        except Exception as exc:
            print(f"[hf_download_tool] Standard split failed: {exc}. Trying first available...")
            ds = load_dataset(dataset_ref)
            keys = list(ds.keys())
            if not keys:
                raise ValueError("Dataset contains no splits.")
            ds = ds[keys[0]]

        df = ds.to_pandas()
        mem_usage = df.memory_usage(deep=True).sum()
        if mem_usage > 500 * 1024 * 1024:
            return _download_result(
                status="unsupported_format",
                source="huggingface",
                dataset_ref=dataset_ref,
                error_message=f"Dataset {dataset_ref} exceeds 500MB limit ({mem_usage / 1024 / 1024:.1f} MB).",
            )

        df.to_csv(csv_path, index=False)
        print(f"[hf_download_tool] Saved CSV to {csv_path} ({mem_usage / 1024 / 1024:.1f} MB)")
        return _download_result(
            status="ok",
            source="huggingface",
            dataset_ref=dataset_ref,
            resolved_path=str(csv_path.absolute()),
            detected_format=_detect_tabular_format(csv_path),
        )
    except Exception as exc:
        error_message = f"HuggingFace download failed for {dataset_ref} - {type(exc).__name__}: {exc}"
        return _download_result(
            status=_classify_download_error(error_message, source="huggingface"),
            source="huggingface",
            dataset_ref=dataset_ref,
            error_message=error_message,
        )


@tool
def arxiv_search_tool(query: str) -> str:
    """
    Search ArXiv Export API for scholarly papers matching the query.
    Returns the top 5 relevant papers (title, summary, date, link).
    """
    import requests
    import xml.etree.ElementTree as ET
    try:
        # ArXiv API setup
        base_url = "http://export.arxiv.org/api/query?"
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": 5,
            "sortBy": "relevance"
        }
        
        response = requests.get(base_url, params=params, timeout=15)
        if response.status_code != 200:
            return f"ERROR: ArXiv API returned status {response.status_code}"
            
        # Parse Atom XML
        root = ET.fromstring(response.content)
        # Atom namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        entries = root.findall('atom:entry', ns)
        if not entries:
            return "ERROR: No relevant research papers found on ArXiv."
            
        results = []
        for entry in entries:
            title = entry.find('atom:title', ns).text.strip().replace("\n", " ")
            summary = entry.find('atom:summary', ns).text.strip().replace("\n", " ")
            published = entry.find('atom:published', ns).text[:10] # YYYY-MM-DD
            
            # Find the PDF link if available
            link = ""
            for l in entry.findall('atom:link', ns):
                if l.get('title') == 'pdf' or 'pdf' in l.get('href', ''):
                    link = l.get('href')
                    break
            if not link:
                link = entry.find('atom:id', ns).text # Fallback to abstract URL
                
            results.append(f"TITLE: {title}\nDATE: {published}\nLINK: {link}\nSUMMARY: {summary[:500]}...")
            
        return "\n---\n".join(results)
        
    except Exception as e:
        return f"ERROR: ArXiv search failed: {e}"
