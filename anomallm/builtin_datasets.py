from __future__ import annotations

from pathlib import Path
from typing import Optional

from anomallm.featurize import (
    BUILTIN_CIFAR_REF,
    BUILTIN_IMDB_REF,
    materialize_builtin,
    resolve_local_upload_ref,
)

BUILTIN_REFS = {BUILTIN_IMDB_REF, BUILTIN_CIFAR_REF}


def is_builtin_ref(ref: str) -> bool:
    return ref.strip().lower() in BUILTIN_REFS or ref.startswith("omniml/")


def is_local_ref(ref: str) -> bool:
    return ref.startswith("local://")


def builtin_download_tool(dataset_ref: str, run_id: Optional[str] = None) -> dict:
    """Materialize omniml/* builtin datasets to a features CSV."""
    try:
        out_dir = Path("runs") / (run_id or "default") / "artifacts"
        meta = materialize_builtin(dataset_ref, str(out_dir))
        return {
            "status": "ok",
            "source": "builtin",
            "dataset_ref": dataset_ref,
            "resolved_path": meta["resolved_path"],
            "detected_format": "csv",
            "error_message": "",
            "featurization": meta,
        }
    except Exception as exc:
        return {
            "status": "download_failed",
            "source": "builtin",
            "dataset_ref": dataset_ref,
            "resolved_path": "",
            "detected_format": "",
            "error_message": str(exc),
        }


def local_upload_download_tool(dataset_ref: str) -> dict:
    """Resolve a Chainlit local:// upload to an on-disk path."""
    try:
        resolved = resolve_local_upload_ref(dataset_ref)
        path = Path(resolved)
        if not path.exists():
            raise FileNotFoundError(f"Upload not found: {resolved}")
        suffix = path.suffix.lower()
        if suffix in {".csv", ".tsv"}:
            kind = "tabular_csv"
            modality = "tabular"
        elif suffix in {".txt", ".jsonl"}:
            kind = "text_raw"
            modality = "text"
        elif suffix in {".zip"} or path.is_dir():
            kind = "image_raw"
            modality = "image"
        elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
            kind = "image_raw"
            modality = "image"
        else:
            kind = "unknown"
            modality = "tabular"
        return {
            "status": "ok",
            "source": "local",
            "dataset_ref": dataset_ref,
            "resolved_path": str(path.resolve()),
            "detected_format": suffix.lstrip(".") or "dir",
            "error_message": "",
            "detected_modality": modality,
            "kind": kind,
        }
    except Exception as exc:
        return {
            "status": "download_failed",
            "source": "local",
            "dataset_ref": dataset_ref,
            "resolved_path": "",
            "detected_format": "",
            "error_message": str(exc),
        }
