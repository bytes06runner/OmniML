from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

MAX_SHAP_ROWS = 500
MAX_BACKGROUND = 50
MAX_LIME_INSTANCES = 5


def encode_tabular_frame(
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, str, Optional[LabelEncoder]]:
    feature_df = df.drop(columns=[target_col]).copy()
    y_raw = df[target_col].copy()
    encoded_features = feature_df.copy()
    for col in encoded_features.columns:
        if str(encoded_features[col].dtype) in ("object", "category", "bool"):
            encoded_features[col] = LabelEncoder().fit_transform(encoded_features[col].astype(str).fillna("missing"))
        else:
            encoded_features[col] = pd.to_numeric(encoded_features[col], errors="coerce")
    encoded_features = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(encoded_features),
        columns=encoded_features.columns,
    )
    task_type = "classification"
    if y_raw.dtype.kind in ("f",) and y_raw.nunique() > 20:
        task_type = "regression"
    target_encoder = None
    if task_type == "classification":
        target_encoder = LabelEncoder()
        y = target_encoder.fit_transform(y_raw.astype(str).fillna("missing"))
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(pd.to_numeric(y_raw, errors="coerce").mean()).values
    return encoded_features, np.asarray(y), task_type, target_encoder


def load_model_and_matrix(
    csv_path: str,
    target_col: Optional[str],
    model_path: str,
    max_rows: int = MAX_SHAP_ROWS,
) -> Tuple[Any, pd.DataFrame, np.ndarray, List[str], str]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    df = pd.read_csv(csv_path, sep=None, engine="python")
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)
    resolved_target = target_col or df.columns[-1]
    X_df, y, task_type, _ = encode_tabular_frame(df, resolved_target)
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)
    return model, X_df, y, X_df.columns.tolist(), task_type


def compute_global_shap(
    model: Any,
    X: pd.DataFrame,
    feature_names: List[str],
    plots_dir: str,
) -> Tuple[Dict[str, Any], Optional[str]]:
    import shap

    os.makedirs(plots_dir, exist_ok=True)
    X_matrix = X.values
    background = X_matrix[: min(MAX_BACKGROUND, len(X_matrix))]
    plot_path = os.path.join(plots_dir, "shap_summary.png")

    try:
        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_matrix)
        else:
            predict_fn = model.predict_proba if hasattr(model, "predict_proba") else model.predict
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_matrix[: min(100, len(X_matrix))])

        if isinstance(shap_values, list):
            values = np.asarray(shap_values[0])
        else:
            values = np.asarray(shap_values)
        mean_abs = np.abs(values).mean(axis=0)
        ranking = [
            {"feature": feature_names[i], "mean_abs_shap": float(mean_abs[i])}
            for i in range(min(len(feature_names), len(mean_abs)))
        ]
        ranking.sort(key=lambda item: item["mean_abs_shap"], reverse=True)

        plt.figure(figsize=(8, 5))
        top = ranking[:10]
        if top:
            plt.barh([row["feature"] for row in top][::-1], [row["mean_abs_shap"] for row in top][::-1])
            plt.xlabel("Mean |SHAP|")
            plt.title("SHAP Global Summary")
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()
        else:
            plot_path = None

        return {"top_features": ranking[:10], "sample_rows": int(values.shape[0])}, plot_path
    except Exception as exc:
        return {"error": str(exc), "top_features": []}, None


def _pick_lime_indices(y: np.ndarray, task_type: str, max_instances: int) -> List[int]:
    indices = list(range(min(max_instances, len(y))))
    if task_type != "classification" or len(y) == 0:
        return indices
    classes = np.unique(y)
    picked: List[int] = []
    for cls in classes:
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx):
            picked.append(int(cls_idx[0]))
    for idx in indices:
        if idx not in picked and len(picked) < max_instances:
            picked.append(idx)
    return picked[:max_instances]


def compute_local_lime(
    model: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_names: List[str],
    task_type: str,
) -> List[Dict[str, Any]]:
    from lime.lime_tabular import LimeTabularExplainer

    X_matrix = X.values.astype(float)
    class_names = None
    mode = "regression"
    if task_type == "classification":
        mode = "classification"
        class_names = [str(label) for label in np.unique(y)]

    explainer = LimeTabularExplainer(
        X_matrix,
        feature_names=feature_names,
        class_names=class_names,
        mode=mode,
        discretize_continuous=True,
    )
    predict_fn = model.predict_proba if hasattr(model, "predict_proba") and task_type == "classification" else model.predict
    explanations: List[Dict[str, Any]] = []
    for idx in _pick_lime_indices(y, task_type, MAX_LIME_INSTANCES):
        try:
            exp = explainer.explain_instance(X_matrix[idx], predict_fn, num_features=min(8, len(feature_names)))
            explanations.append(
                {
                    "row_index": int(idx),
                    "label": int(y[idx]) if task_type == "classification" else float(y[idx]),
                    "features": [
                        {"feature": name, "weight": float(weight)}
                        for name, weight in exp.as_list()
                    ],
                }
            )
        except Exception as exc:
            explanations.append({"row_index": int(idx), "error": str(exc)})
    return explanations


def run_tabular_xai(
    csv_path: str,
    target_col: Optional[str],
    model_path: str,
    plots_dir: str,
    artifacts_dir: str,
) -> Dict[str, Any]:
    limitations: List[str] = []
    plot_paths: Dict[str, str] = {}
    global_shap: Dict[str, Any] = {}
    local_lime: List[Dict[str, Any]] = []
    top_features: List[Dict[str, Any]] = []
    status = "limited"
    explanation_method = "heuristic_fallback"

    try:
        model, X_df, y, feature_names, task_type = load_model_and_matrix(csv_path, target_col, model_path)
        if task_type != "classification":
            limitations.append("LIME local explanations are only generated for classification tasks in this release.")

        global_shap, shap_plot = compute_global_shap(model, X_df, feature_names, plots_dir)
        if shap_plot:
            plot_paths["shap_summary"] = shap_plot
            top_features = [
                {"feature": row["feature"], "importance": row["mean_abs_shap"]}
                for row in global_shap.get("top_features", [])
            ]
            explanation_method = "shap_global"
            status = "generated"

        if task_type == "classification":
            local_lime = compute_local_lime(model, X_df, y, feature_names, task_type)
            lime_path = os.path.join(artifacts_dir, "local_lime.json")
            os.makedirs(artifacts_dir, exist_ok=True)
            with open(lime_path, "w", encoding="utf-8") as handle:
                json.dump(local_lime, handle, indent=2, default=str)
            plot_paths["local_lime"] = lime_path
            if local_lime:
                explanation_method = "shap_global+lime_local" if shap_plot else "lime_local"
                status = "generated"

        if global_shap.get("error"):
            limitations.append(f"SHAP computation partial failure: {global_shap['error']}")
    except Exception as exc:
        limitations.append(str(exc))
        status = "limited"

    narrative = (
        "Global SHAP and local LIME explanations were generated from the trained tabular model when supported. "
        "Interpret alongside dataset quality checks and fairness evidence."
        if status == "generated"
        else "Explainability is limited for this run; review feature importance plots and model card metadata."
    )

    return {
        "status": status,
        "explanation_method": explanation_method,
        "narrative": narrative,
        "top_features": top_features,
        "global_shap": global_shap,
        "local_lime": local_lime,
        "plot_paths": plot_paths,
        "limitations": limitations,
        "modality": "tabular",
    }


def _is_sklearn_pickle(model_path: str) -> bool:
    if not model_path or not os.path.exists(model_path):
        return False
    try:
        with open(model_path, "rb") as handle:
            obj = pickle.load(handle)
        return hasattr(obj, "predict")
    except Exception:
        return False


def _tag_explanation_method(payload: Dict[str, Any], suffix: str, modality: str) -> Dict[str, Any]:
    method = payload.get("explanation_method", "heuristic_fallback")
    if method == "heuristic_fallback":
        payload["explanation_method"] = suffix
    elif suffix not in method:
        payload["explanation_method"] = f"{method}_{suffix}"
    payload["modality"] = modality
    return payload


def run_text_xai(
    csv_path: str,
    target_col: Optional[str],
    model_path: str,
    plots_dir: str,
    artifacts_dir: str,
) -> Dict[str, Any]:
    limitations = [
        "Text modality explains TF-IDF or numeric feature columns from the exported CSV, not raw tokens.",
        "IMDB-style offline runs use a 20newsgroups proxy; token-level LIME requires a raw text column in the dataset.",
    ]
    if not csv_path or not os.path.exists(csv_path):
        return {
            "status": "limited",
            "explanation_method": "text_tfidf_limited",
            "narrative": "Text explainability requires a tabular feature export (CSV) and a trained sklearn model.",
            "top_features": [],
            "global_shap": {},
            "local_lime": [],
            "plot_paths": {},
            "limitations": limitations + ["Dataset CSV not available."],
            "modality": "text",
        }
    if not _is_sklearn_pickle(model_path):
        return {
            "status": "limited",
            "explanation_method": "text_tfidf_limited",
            "narrative": "Text explainability requires a pickled sklearn model export.",
            "top_features": [],
            "global_shap": {},
            "local_lime": [],
            "plot_paths": {},
            "limitations": limitations + [f"Sklearn model not found at {model_path}"],
            "modality": "text",
        }
    payload = run_tabular_xai(csv_path, target_col, model_path, plots_dir, artifacts_dir)
    payload["limitations"] = list(dict.fromkeys((payload.get("limitations") or []) + limitations))
    payload["narrative"] = (
        "Global SHAP and local LIME were computed on text-derived tabular features (e.g. TF-IDF). "
        "Interpret alongside dataset quality checks and fairness evidence."
        if payload.get("status") == "generated"
        else payload.get("narrative", "")
    )
    return _tag_explanation_method(payload, "text_tfidf", "text")


def run_image_xai(
    csv_path: str,
    target_col: Optional[str],
    model_path: str,
    plots_dir: str,
    artifacts_dir: str,
) -> Dict[str, Any]:
    limitations = [
        "Image UI training is not fully implemented; explainability targets flattened-pixel sklearn exports when available.",
        "For paper-aligned CIFAR-10 metrics, run: python -m experiments.run --dataset cifar10",
    ]
    if not csv_path or not os.path.exists(csv_path):
        return {
            "status": "limited",
            "explanation_method": "image_limited",
            "narrative": "Image explainability is limited without a flattened feature CSV and sklearn model.",
            "top_features": [],
            "global_shap": {},
            "local_lime": [],
            "plot_paths": {},
            "limitations": limitations,
            "modality": "image",
        }
    if not _is_sklearn_pickle(model_path):
        return {
            "status": "limited",
            "explanation_method": "image_limited",
            "narrative": (
                "PyTorch-only model exports cannot be explained with the current SHAP/LIME path. "
                "Use offline CIFAR-10 experiments or a sklearn model on flattened features."
            ),
            "top_features": [],
            "global_shap": {},
            "local_lime": [],
            "plot_paths": {},
            "limitations": limitations + [f"No sklearn pickle at {model_path}"],
            "modality": "image",
        }
    payload = run_tabular_xai(csv_path, target_col, model_path, plots_dir, artifacts_dir)
    payload["limitations"] = list(dict.fromkeys((payload.get("limitations") or []) + limitations))
    return _tag_explanation_method(payload, "image_flattened", "image")


def load_model_meta(model_path: str) -> Dict[str, Any]:
    if not model_path:
        return {}
    meta_path = os.path.join(os.path.dirname(model_path), "model_meta.json")
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def is_pytorch_export(model_path: str) -> bool:
    meta = load_model_meta(model_path)
    if meta.get("engineer_template_id") == "pytorch_mlp" or meta.get("training_path") == "pytorch":
        return True
    if meta.get("model_format") == "state_dict":
        return True
    return not _is_sklearn_pickle(model_path) and os.path.exists(model_path)


def run_pytorch_limited_xai(modality: str, model_path: str, artifacts_dir: str) -> Dict[str, Any]:
    summary_path = os.path.join(artifacts_dir, "xai_summary.json")
    limitations = [
        "Path A PyTorch training: model export is a torch state_dict, not a sklearn pickle.",
        "Global SHAP and local LIME in the Chainlit pipeline require sklearn or a DeepExplainer integration.",
        "Review loss curves, evaluation.json, and model_card.json for this run.",
    ]
    payload = {
        "status": "limited",
        "explanation_method": "pytorch_limited",
        "narrative": (
            "Explainability is limited for PyTorch architect-graph runs. "
            "Training artifacts and fairness evidence should be used for trust review."
        ),
        "top_features": [],
        "global_shap": {},
        "local_lime": [],
        "plot_paths": {},
        "limitations": limitations,
        "modality": modality,
        "model_path": model_path,
    }
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
    payload["plot_paths"] = {"xai_summary": summary_path}
    return payload


def run_xai_for_modality(
    modality: str,
    csv_path: str,
    target_col: Optional[str],
    model_path: str,
    plots_dir: str,
    artifacts_dir: str,
) -> Dict[str, Any]:
    normalized = (modality or "tabular").lower()
    if is_pytorch_export(model_path):
        return run_pytorch_limited_xai(normalized, model_path, artifacts_dir)
    if normalized == "text":
        return run_text_xai(csv_path, target_col, model_path, plots_dir, artifacts_dir)
    if normalized == "image":
        return run_image_xai(csv_path, target_col, model_path, plots_dir, artifacts_dir)
    payload = run_tabular_xai(csv_path, target_col, model_path, plots_dir, artifacts_dir)
    payload["modality"] = "tabular"
    return payload
