from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
BUILTIN_IMDB_REF = "omniml/imdb-text-proxy"
BUILTIN_CIFAR_REF = "omniml/cifar10-image-proxy"


def build_imdb_proxy_frame(sample_size: int = 3000, max_features: int = 200) -> pd.DataFrame:
    """20newsgroups binary proxy + TF-IDF (offline/Chainlit text demo)."""
    categories = ["rec.sport.baseball", "sci.space"]
    train = fetch_20newsgroups(subset="train", categories=categories, shuffle=True, random_state=42)
    texts = train.data[:sample_size]
    labels = train.target[:sample_size]
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    frame = pd.DataFrame(matrix.toarray(), columns=[f"t{i}" for i in range(matrix.shape[1])])
    frame["target"] = labels
    return frame


def build_cifar_proxy_frame(sample_size: int = 2000, random_state: int = 42) -> pd.DataFrame:
    try:
        import torchvision
        from torchvision import transforms
    except ImportError as exc:
        raise ImportError("torchvision is required for CIFAR-10. Install with: pip install torchvision") from exc

    dataset = torchvision.datasets.CIFAR10(
        root="./data/cifar10",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    n = min(sample_size, len(dataset))
    indices = np.random.default_rng(random_state).choice(len(dataset), size=n, replace=False)
    rows = []
    for idx in indices:
        tensor, label = dataset[int(idx)]
        flat = tensor.numpy().reshape(-1)
        rows.append((*flat.tolist(), int(label)))
    n_features = len(rows[0]) - 1 if rows else 3072
    columns = [f"p{i}" for i in range(n_features)] + ["target"]
    return pd.DataFrame(rows, columns=columns)


def _read_text_samples(path: Path) -> Tuple[List[str], List[Any]]:
    suffix = path.suffix.lower()
    texts: List[str] = []
    labels: List[Any] = []

    if suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                text = row.get("text") or row.get("content") or row.get("review") or ""
                label = row.get("label", row.get("target", row.get("sentiment", 0)))
                texts.append(str(text))
                labels.append(label)
    elif suffix == ".txt":
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if "\t" in line:
                    label, text = line.split("\t", 1)
                    labels.append(label.strip())
                    texts.append(text.strip())
                else:
                    texts.append(line)
        if texts and not labels:
            raise ValueError(
                "Plain .txt uploads need tab-separated 'label<TAB>text' lines, or use .jsonl with text/label fields."
            )
    elif suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path, sep=None, engine="python")
        text_col = next((c for c in df.columns if str(c).lower() in {"text", "content", "review", "body"}), None)
        target_col = next((c for c in df.columns if str(c).lower() in {"target", "label", "sentiment"}), df.columns[-1])
        if text_col is None:
            text_col = df.columns[0]
        texts = df[text_col].astype(str).tolist()
        labels = df[target_col].tolist()
    else:
        raise ValueError(f"Unsupported text artifact format: {suffix}")

    if not texts:
        raise ValueError("No text samples found in upload.")
    return texts, labels


def featurize_text(raw_path: str, out_csv: str, max_features: int = 200) -> Dict[str, Any]:
    path = Path(raw_path)
    texts, labels = _read_text_samples(path)
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    frame = pd.DataFrame(matrix.toarray(), columns=[f"t{i}" for i in range(matrix.shape[1])])
    frame["target"] = labels
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return {
        "modality": "text",
        "target_column": "target",
        "feature_columns": [c for c in frame.columns if c != "target"],
        "row_count": len(frame),
        "source": str(path),
        "limitations": [
            "Text featurization uses TF-IDF on uploaded content (not raw-token LIME).",
            "IMDB builtin uses a 20newsgroups proxy dataset.",
        ],
    }


def _collect_image_paths(path: Path) -> List[Tuple[Path, str]]:
    items: List[Tuple[Path, str]] = []
    if path.is_file() and path.suffix.lower() == ".zip":
        extract_dir = path.parent / f"{path.stem}_extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(extract_dir)
        path = extract_dir

    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [(path, "0")]

    if path.is_dir():
        image_files = [
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if not image_files:
            raise ValueError(f"No images found under {path}")
        # Class = immediate parent folder name when nested
        for img in image_files:
            label = img.parent.name if img.parent != path else "0"
            items.append((img, label))
    else:
        raise ValueError(f"Image path is not a directory, zip, or image file: {path}")

    return items


def _load_image_flat(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        rgb = img.convert("RGB").resize((32, 32))
        return np.asarray(rgb, dtype=np.float32).reshape(-1) / 255.0


def featurize_image_dir(
    raw_path: str,
    out_csv: str,
    sample_size: int = 2000,
    random_state: int = 42,
) -> Dict[str, Any]:
    path = Path(raw_path)
    pairs = _collect_image_paths(path)
    if len(pairs) > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(pairs), size=sample_size, replace=False)
        pairs = [pairs[int(i)] for i in idx]

    label_encoder = {label: i for i, label in enumerate(sorted({lbl for _, lbl in pairs}))}
    rows = []
    for img_path, label in pairs:
        flat = _load_image_flat(img_path)
        rows.append((*flat.tolist(), label_encoder[label]))

    n_features = len(rows[0]) - 1 if rows else 3072
    columns = [f"p{i}" for i in range(n_features)] + ["target"]
    frame = pd.DataFrame(rows, columns=columns)
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    return {
        "modality": "image",
        "target_column": "target",
        "feature_columns": [c for c in frame.columns if c != "target"],
        "row_count": len(frame),
        "source": str(path),
        "limitations": [
            "Images are resized to 32x32 and flattened for sklearn Path B (not CNN training).",
        ],
    }


def is_featurized_csv(path: str) -> bool:
    """True if CSV already looks like TF-IDF or flattened-pixel features."""
    try:
        sample = pd.read_csv(path, nrows=5)
    except Exception:
        return False
    cols = [str(c) for c in sample.columns]
    if "target" not in cols:
        return False
    feature_cols = [c for c in cols if c != "target"]
    if not feature_cols:
        return False
    return all(c.startswith("t") or c.startswith("p") for c in feature_cols)


def materialize_builtin(ref: str, out_dir: str) -> Dict[str, Any]:
    out_path = Path(out_dir) / "features.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    normalized = ref.strip().lower()
    if normalized in {BUILTIN_IMDB_REF, "omniml/imdb-text-proxy"}:
        frame = build_imdb_proxy_frame()
        limitation = "Builtin IMDB proxy uses 20newsgroups TF-IDF, not the HuggingFace IMDB set."
        modality = "text"
    elif normalized in {BUILTIN_CIFAR_REF, "omniml/cifar10-image-proxy"}:
        frame = build_cifar_proxy_frame()
        limitation = "Builtin CIFAR proxy uses flattened pixels + sklearn, not in-app CNN training."
        modality = "image"
    else:
        raise ValueError(f"Unknown builtin dataset ref: {ref}")

    frame.to_csv(out_path, index=False)
    return {
        "modality": modality,
        "target_column": "target",
        "feature_columns": [c for c in frame.columns if c != "target"],
        "row_count": len(frame),
        "resolved_path": str(out_path),
        "source": ref,
        "limitations": [limitation],
    }


def resolve_local_upload_ref(ref: str) -> str:
    """Map local://data/uploads/... to filesystem path."""
    if ref.startswith("local://"):
        return str(Path(ref[len("local://") :]))
    return ref
