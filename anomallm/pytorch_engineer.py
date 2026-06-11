from __future__ import annotations

import json
import textwrap
from typing import Any, Dict, List

from anomallm.graph_compile import build_mlp_layers
from anomallm.hpo import merge_approved_params, pytorch_search_space


def _indent_block(block: str, extra_spaces: int = 8) -> str:
    pad = " " * extra_spaces
    return "\n".join(pad + line for line in block.splitlines())


def _default_pytorch_hpt(training_config: Dict[str, Any]) -> Dict[str, Any]:
    candidates = pytorch_search_space([], training_config).get("candidates") or []
    if candidates:
        first = candidates[0]
        return {"kind": first.get("kind", "pytorch"), "params": first.get("params") or {}}
    return {
        "kind": "pytorch",
        "params": {
            "learning_rate": float(training_config.get("learning_rate") or training_config.get("lr") or 0.001),
            "batch_size": int(training_config.get("batch_size") or 64),
        },
    }


def generate_pytorch_training_script(state: Dict[str, Any]) -> str:
    run_manifest = state.get("run_manifest") or {}
    run_id = run_manifest.get("run_id", state.get("problem_id", "run_default"))
    training_config = state.get("training_config") or {}
    fairness_config = training_config.get("fairness_config") or {}
    compliance_modes = training_config.get("compliance_modes") or ["eu_ai_act", "fda_samd", "soc2"]
    imbalance = state.get("imbalance") or {}
    modality_label = state.get("modality") or (state.get("task_representation") or {}).get("modality") or "tabular"

    graph_json = state.get("graph_architecture_json") or {}
    graph_nodes = graph_json.get("nodes") or []
    init_body, forward_body = build_mlp_layers(graph_nodes)
    init_body = _indent_block(init_body, 8)
    forward_body = _indent_block(forward_body, 8)

    hpt_approved = None
    if state.get("is_hpt_approved"):
        hpt_approved = merge_approved_params(state.get("hpt_approved_params"), state.get("theta_star"))
    elif state.get("hpt_approved_params"):
        hpt_approved = state.get("hpt_approved_params")
    if not hpt_approved:
        hpt_approved = _default_pytorch_hpt(training_config)

    hpt_candidates = pytorch_search_space(graph_nodes, training_config).get("candidates") or []

    script = textwrap.dedent(
        f"""
        import json, os, pickle
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        RUN_ID = {run_id!r}
        DATASET_CSV_PATH = {state.get("dataset_csv_path", "")!r}
        USER_QUERY = {state.get("user_query", "")!r}
        TRAINING_CONFIG = {training_config!r}
        FAIRNESS_CONFIG = {fairness_config!r}
        COMPLIANCE_MODES = {compliance_modes!r}
        IMBALANCE = {imbalance!r}
        ENGINEER_TEMPLATE_ID = "pytorch_mlp"
        TRAINING_PATH = "pytorch"
        MODALITY = {modality_label!r}
        HPT_APPROVED = {hpt_approved!r}
        HPT_CANDIDATES = {hpt_candidates!r}
        FORCE_HPT = {bool(state.get("is_hpt_approved"))!r}
        CWD = os.getcwd()
        RUN_ROOT = os.path.join(CWD, "runs", RUN_ID)
        ARTIFACTS_DIR = os.path.join(RUN_ROOT, "artifacts")
        PLOTS_DIR = os.path.join(RUN_ROOT, "plots")
        REPORTS_DIR = os.path.join(RUN_ROOT, "reports")
        EXPORTS_DIR = os.path.join(RUN_ROOT, "exports")
        LOGS_DIR = os.path.join(RUN_ROOT, "logs")
        LEGACY_EXPORTS = os.path.join(CWD, "exports")
        for _path in (RUN_ROOT, ARTIFACTS_DIR, PLOTS_DIR, REPORTS_DIR, EXPORTS_DIR, LOGS_DIR, LEGACY_EXPORTS):
            os.makedirs(_path, exist_ok=True)

        df = pd.read_csv(DATASET_CSV_PATH, sep=None, engine="python")
        target_col = TRAINING_CONFIG.get("target_column") or df.columns[-1]
        feature_df = df.drop(columns=[target_col]).copy()
        y_raw = df[target_col].copy()
        sensitive_candidates = [col for col in df.columns if any(token in col.lower() for token in ("gender","sex","race","ethnicity","age","religion","disability","marital","nationality"))]

        encoded_features = feature_df.copy()
        for col in encoded_features.columns:
            if str(encoded_features[col].dtype) in ("object", "category", "bool"):
                encoded_features[col] = LabelEncoder().fit_transform(encoded_features[col].astype(str).fillna("missing"))
            else:
                encoded_features[col] = pd.to_numeric(encoded_features[col], errors="coerce")

        encoded_features = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(encoded_features), columns=encoded_features.columns)
        y_series = y_raw.copy()
        task_type = "classification"
        if y_series.dtype.kind in ("f",) and y_series.nunique() > 20:
            task_type = "regression"

        target_encoder = None
        if task_type == "classification":
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y_series.astype(str).fillna("missing"))
            num_classes = int(len(np.unique(y)))
        else:
            y = pd.to_numeric(y_series, errors="coerce").fillna(pd.to_numeric(y_series, errors="coerce").mean()).values
            num_classes = 1

        input_dim = int(encoded_features.shape[1])
        stratify = y if task_type == "classification" and len(set(y)) > 1 else None
        test_size = float(TRAINING_CONFIG.get("test_size") or 0.2)
        X_train, X_val, y_train, y_val, idx_train, idx_val = train_test_split(
            encoded_features, y, np.arange(len(encoded_features)), test_size=test_size, random_state=42, stratify=stratify
        )

        applied_strategy = None
        train_sample_weight = None
        if task_type == "classification":
            strategy = IMBALANCE.get("recommended_strategy", "balanced")
            if strategy == "adasyn":
                try:
                    from imblearn.over_sampling import ADASYN
                    X_train, y_train = ADASYN(random_state=42).fit_resample(X_train, y_train)
                    applied_strategy = "adasyn"
                except Exception as exc:
                    try:
                        from imblearn.over_sampling import SMOTE
                        X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
                        applied_strategy = "smote"
                        IMBALANCE["warnings"] = list(IMBALANCE.get("warnings", [])) + [f"ADASYN failed: {{exc}}; fell back to SMOTE"]
                    except Exception as exc2:
                        applied_strategy = "class_weight"
                        IMBALANCE["warnings"] = list(IMBALANCE.get("warnings", [])) + [f"ADASYN failed: {{exc}}", f"SMOTE fallback failed: {{exc2}}"]
            elif strategy == "smote":
                try:
                    from imblearn.over_sampling import SMOTE
                    X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
                    applied_strategy = "smote"
                except Exception as exc:
                    applied_strategy = "class_weight"
                    IMBALANCE["warnings"] = list(IMBALANCE.get("warnings", [])) + [f"SMOTE failed: {{exc}}"]
            elif strategy == "focal":
                counts = np.bincount(np.asarray(y_train, dtype=int))
                counts = np.maximum(counts, 1)
                p_class = counts[np.asarray(y_train, dtype=int)] / float(counts.sum())
                train_sample_weight = (1.0 / p_class) ** 2.0
                train_sample_weight = train_sample_weight * (len(train_sample_weight) / train_sample_weight.sum())
                applied_strategy = "focal"
            elif strategy == "class_weight":
                applied_strategy = "class_weight"
            else:
                applied_strategy = "balanced"

        class OmniMLNet(nn.Module):
            def __init__(self, input_dim, num_classes):
                super().__init__()
{init_body}

            def forward(self, x):
{forward_body}
                return x

        def _make_loader(X_df, y_arr, batch_size, shuffle, weights=None):
            X_t = torch.tensor(X_df.values, dtype=torch.float32)
            if task_type == "classification":
                y_t = torch.tensor(y_arr, dtype=torch.long)
            else:
                y_t = torch.tensor(y_arr, dtype=torch.float32)
            if weights is not None:
                w_t = torch.tensor(weights, dtype=torch.float32)
                ds = TensorDataset(X_t, y_t, w_t)
            else:
                ds = TensorDataset(X_t, y_t)
            return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

        def _build_optimizer(model, name, lr):
            name = (name or "adam").lower()
            if name == "sgd":
                return torch.optim.SGD(model.parameters(), lr=lr)
            if name == "rmsprop":
                return torch.optim.RMSprop(model.parameters(), lr=lr)
            if name == "adamw":
                return torch.optim.AdamW(model.parameters(), lr=lr)
            return torch.optim.Adam(model.parameters(), lr=lr)

        def _train_eval(model, train_loader, val_X, val_y, epochs, criterion, optimizer, sample_weights=None):
            val_X_t = torch.tensor(val_X.values, dtype=torch.float32)
            if task_type == "classification":
                val_y_t = torch.tensor(val_y, dtype=torch.long)
            else:
                val_y_t = torch.tensor(val_y, dtype=torch.float32)
            epoch_metrics = []
            for epoch in range(1, epochs + 1):
                model.train()
                running_loss = 0.0
                n_batches = 0
                correct = 0
                total = 0
                for batch in train_loader:
                    optimizer.zero_grad()
                    if len(batch) == 3:
                        xb, yb, wb = batch
                    else:
                        xb, yb = batch
                        wb = None
                    out = model(xb)
                    if task_type == "classification":
                        loss_vec = criterion(out, yb)
                        loss = (loss_vec * wb).mean() if wb is not None else loss_vec.mean()
                        preds = out.argmax(dim=1)
                        correct += int((preds == yb).sum().item())
                        total += int(yb.size(0))
                    else:
                        loss_vec = criterion(out.squeeze(-1), yb)
                        loss = (loss_vec * wb).mean() if wb is not None else loss_vec.mean()
                    loss.backward()
                    optimizer.step()
                    running_loss += float(loss.item())
                    n_batches += 1
                model.eval()
                with torch.no_grad():
                    val_out = model(val_X_t)
                    if task_type == "classification":
                        val_loss = float(criterion(val_out, val_y_t).mean().item())
                        val_preds = val_out.argmax(dim=1).cpu().numpy()
                        val_acc = float(accuracy_score(val_y, val_preds))
                        train_acc = float(correct / max(total, 1))
                        metric = {{
                            "epoch": epoch,
                            "loss": running_loss / max(n_batches, 1),
                            "val_loss": val_loss,
                            "acc": train_acc,
                            "val_acc": val_acc,
                        }}
                    else:
                        val_loss = float(criterion(val_out.squeeze(-1), val_y_t).mean().item())
                        metric = {{
                            "epoch": epoch,
                            "loss": running_loss / max(n_batches, 1),
                            "val_loss": val_loss,
                            "acc": 0.0,
                            "val_acc": 0.0,
                        }}
                epoch_metrics.append(metric)
                print(json.dumps({{"type": "epoch_metric", **metric}}), flush=True)
            return epoch_metrics

        def _run_trial(params):
            lr = float(params.get("learning_rate") or params.get("lr") or TRAINING_CONFIG.get("learning_rate") or TRAINING_CONFIG.get("lr") or 0.001)
            batch_size = int(params.get("batch_size") or TRAINING_CONFIG.get("batch_size") or 64)
            epochs = int(TRAINING_CONFIG.get("epochs") or 10)
            optimizer_name = TRAINING_CONFIG.get("optimizer") or "adam"
            torch.manual_seed(int(TRAINING_CONFIG.get("seed") or 42))
            model = OmniMLNet(input_dim, num_classes if task_type == "classification" else 1)
            if task_type == "classification":
                criterion = nn.CrossEntropyLoss(reduction="none")
            else:
                criterion = nn.MSELoss(reduction="none")
            optimizer = _build_optimizer(model, optimizer_name, lr)
            weights = train_sample_weight if applied_strategy == "focal" else None
            train_loader = _make_loader(
                X_train, y_train, batch_size, bool(TRAINING_CONFIG.get("shuffle", True)), weights
            )
            metrics = _train_eval(model, train_loader, X_val, y_val, epochs, criterion, optimizer, weights)
            if task_type == "classification":
                score = float(metrics[-1]["val_acc"]) if metrics else 0.0
            else:
                score = -float(metrics[-1]["val_loss"]) if metrics else 0.0
            return model, metrics, score, {{"learning_rate": lr, "batch_size": batch_size}}

        candidates = []
        if FORCE_HPT and (HPT_APPROVED.get("params") or HPT_APPROVED.get("kind")):
            params = HPT_APPROVED.get("params") or HPT_APPROVED
            if isinstance(params, dict) and "kind" in params and len(params) == 1:
                params = {{}}
            candidates = [params if isinstance(params, dict) else {{}}]
        else:
            for item in HPT_CANDIDATES:
                candidates.append(item.get("params") or {{}})
            if not candidates:
                candidates = [
                    {{"learning_rate": 0.001, "batch_size": 64}},
                    {{"learning_rate": 0.01, "batch_size": 32}},
                    {{"learning_rate": 0.001, "batch_size": 32}},
                ]

        best_score = None
        best_params = None
        trial_results = []
        total = len(candidates)
        for trial_idx, params in enumerate(candidates, start=1):
            model, metrics, score, resolved = _run_trial(params)
            if best_score is None or score > best_score:
                best_score, best_params = score, resolved
            trial_results.append({{"trial": trial_idx, "kind": "pytorch", "params": resolved, "value": float(score)}})
            print(json.dumps({{"type": "hpt_trial", "trial": trial_idx, "total": total, "value": float(score), "best_so_far": float(best_score), "params": resolved}}), flush=True)
        print(json.dumps({{"type": "hpt_complete", "best_params": best_params or {{}}, "best_kind": "pytorch", "best_value": float(best_score or 0.0)}}), flush=True)

        final_model, epoch_metrics, _, _ = _run_trial(best_params or {{}})
        with torch.no_grad():
            val_X_t = torch.tensor(X_val.values, dtype=torch.float32)
            val_out = final_model(val_X_t)
            if task_type == "classification":
                predictions = val_out.argmax(dim=1).cpu().numpy()
                accuracy = float(accuracy_score(y_val, predictions))
                f1 = float(f1_score(y_val, predictions, average="weighted", zero_division=0))
                metrics = {{"task_type": "classification", "accuracy": accuracy, "f1": f1, "val_acc": accuracy, "best_kind": "pytorch"}}
            else:
                predictions = val_out.squeeze(-1).cpu().numpy()
                rmse = float(mean_squared_error(y_val, predictions) ** 0.5)
                r2 = float(r2_score(y_val, predictions))
                metrics = {{"task_type": "regression", "rmse": rmse, "r2": r2, "best_kind": "pytorch"}}

        y_val_human = target_encoder.inverse_transform(np.asarray(y_val, dtype=int)) if task_type == "classification" and target_encoder is not None else y_val
        pred_human = target_encoder.inverse_transform(np.asarray(predictions, dtype=int)) if task_type == "classification" and target_encoder is not None else predictions
        pd.DataFrame([{{"row_index": int(i), "y_true": truth, "y_pred": pred}} for i, truth, pred in zip(idx_val, y_val_human, pred_human)]).to_csv(os.path.join(ARTIFACTS_DIR, "predictions.csv"), index=False)

        feature_importance = []
        imbalance_record = dict(IMBALANCE)
        imbalance_record["applied_strategy"] = applied_strategy
        evaluation = {{
            "metrics": metrics,
            "trial_results": trial_results,
            "target_column": target_col,
            "feature_columns": encoded_features.columns.tolist(),
            "feature_importance": feature_importance,
            "sensitive_candidates": sensitive_candidates,
            "imbalance": imbalance_record,
            "training_path": "pytorch",
        }}
        with open(os.path.join(ARTIFACTS_DIR, "evaluation.json"), "w", encoding="utf-8") as handle:
            json.dump(evaluation, handle, indent=2, default=str)
        with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, default=str)

        model_limitations = [
            f"Training template: {{ENGINEER_TEMPLATE_ID}} (modality={{MODALITY}}).",
            "Path A PyTorch execution: approved architect graph compiled to OmniMLNet.",
            "Global SHAP/LIME in the UI pipeline require sklearn exports; review model_card and evaluation.json.",
            "Text/image runs use featurized CSV inputs (TF-IDF or flattened pixels).",
        ]
        with open(os.path.join(ARTIFACTS_DIR, "model_card.json"), "w", encoding="utf-8") as handle:
            json.dump({{
                "run_id": RUN_ID,
                "user_query": USER_QUERY,
                "task_type": task_type,
                "target_column": target_col,
                "feature_names": encoded_features.columns.tolist(),
                "top_features": feature_importance,
                "engineer_template_id": ENGINEER_TEMPLATE_ID,
                "training_path": TRAINING_PATH,
                "modality": MODALITY,
                "limitations": model_limitations,
            }}, handle, indent=2, default=str)

        plt.figure(figsize=(8, 5))
        plt.plot([m["epoch"] for m in epoch_metrics], [m["loss"] for m in epoch_metrics], label="train_loss")
        plt.plot([m["epoch"] for m in epoch_metrics], [m["val_loss"] for m in epoch_metrics], label="val_loss")
        plt.legend()
        plt.title("Training Curves")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "loss_curve.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        if trial_results:
            plt.plot([item["trial"] for item in trial_results], [item["value"] for item in trial_results], marker="o")
            plt.title("PyTorch HPT History")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"))
        plt.close()

        plt.figure(figsize=(8, 6))
        sns.heatmap(encoded_features.corr().fillna(0).iloc[: min(10, encoded_features.shape[1]), : min(10, encoded_features.shape[1])], cmap="Blues")
        plt.title("Feature Correlation")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "telemetry_distribution.png"))
        plt.close()

        torch.save(final_model.state_dict(), os.path.join(EXPORTS_DIR, "model.pt"))
        with open(os.path.join(EXPORTS_DIR, "model_scripted.pt"), "wb") as handle:
            pickle.dump(final_model, handle)
        with open(os.path.join(EXPORTS_DIR, "model.onnx"), "wb") as handle:
            handle.write(b"omniml-placeholder-onnx")
        with open(os.path.join(EXPORTS_DIR, "model_meta.json"), "w", encoding="utf-8") as handle:
            json.dump({{
                "run_id": RUN_ID,
                "input_dim": input_dim,
                "num_classes": num_classes,
                "feature_names": encoded_features.columns.tolist(),
                "target_col": target_col,
                "final_val_acc": float(metrics.get("val_acc", 0.0)),
                "task_type": task_type,
                "engineer_template_id": ENGINEER_TEMPLATE_ID,
                "training_path": TRAINING_PATH,
                "model_format": "state_dict",
                "fairness_config": FAIRNESS_CONFIG,
                "compliance_modes": COMPLIANCE_MODES,
            }}, handle, indent=2, default=str)
        with open(os.path.join(EXPORTS_DIR, "serve_api.py"), "w", encoding="utf-8") as handle:
            handle.write("from fastapi import FastAPI\\napp=FastAPI()\\n@app.get('/health')\\ndef health(): return {{'ok': True}}\\n")
        with open(os.path.join(EXPORTS_DIR, "requirements.txt"), "w", encoding="utf-8") as handle:
            handle.write("fastapi\\nuvicorn\\npandas\\nscikit-learn\\ntorch\\npydantic\\n")
        with open(os.path.join(EXPORTS_DIR, "Dockerfile"), "w", encoding="utf-8") as handle:
            handle.write("FROM python:3.12-slim\\nWORKDIR /app\\nCOPY requirements.txt .\\nRUN pip install -r requirements.txt\\nCOPY . .\\nCMD ['python','-m','http.server','8080']\\n")

        for name in ("model.pt", "model_scripted.pt", "model.onnx", "model_meta.json", "serve_api.py", "requirements.txt", "Dockerfile"):
            src = os.path.join(EXPORTS_DIR, name)
            dst = os.path.join(LEGACY_EXPORTS, name)
            with open(src, "rb") as src_handle, open(dst, "wb") as dst_handle:
                dst_handle.write(src_handle.read())
        """
    ).strip()
    return script
