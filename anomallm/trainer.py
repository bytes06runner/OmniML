import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from typing import Dict, Any, Optional, Tuple

# Import the AutoScaler from the sibling module
try:
    from .detector import AutoScaler
except ImportError:
    from detector import AutoScaler

# Nuclear Import Hack: Ensure the venv site-packages are seen even in strange subprocess environments
import sys
venv_pkg_path = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'lib', f'python{sys.version_info.major}.{sys.version_info.minor}', 'site-packages')
if venv_pkg_path not in sys.path:
    sys.path.insert(0, venv_pkg_path)

# Now safe to import large libraries
import xgboost as xgb

class OmniTrainer:
    """
    Industrial-grade Auto-ML Training Engine for OmniML.
    Handles data-blind preprocessing via AutoScaler, Optuna-based hyper-tuning,
    and ultra-fast demo execution (5 epochs) for competitions.
    """
    def __init__(self, df: pd.DataFrame, target_col: str, architecture: str = "auto"):
        self.df = df.copy()
        self.target_col = target_col
        self.architecture = architecture.lower()
        self.study = None
        self.best_params = {}
        self.task_type = "regression"
        self.X = None
        self.y = None
        self.scaler = AutoScaler() # Uses the industrial AutoScaler skill
        self.label_encoder = LabelEncoder()
        
        self._detect_task()
        self._preprocess()

    def _detect_task(self):
        """Identify task type and clean the target column."""
        y_raw = self.df[self.target_col].dropna()
        unique_vals = y_raw.nunique()
        is_float = y_raw.dtype in ['float64', 'float32']
        
        if unique_vals <= 10 or (not is_float and unique_vals < 50):
            self.task_type = "classification"
            print(f"[trainer] 🎯 Task: Classification ({unique_vals} classes)")
        else:
            self.task_type = "regression"
            print(f"[trainer] 📈 Task: Regression")

    def _preprocess(self):
        """Clean features using OmniML AutoScaler and encode target."""
        y_raw = self.df[self.target_col]
        X_raw = self.df.drop(columns=[self.target_col])
        
        # 1. Feature Discovery via AutoScaler (Always use industrial skill)
        self.X = self.scaler.fit_transform(X_raw)
        print(f"[trainer] 📐 AutoScaler identified {len(self.scaler.active_columns)} active features.")
        
        # 2. Target Encoding
        if self.task_type == "classification":
            fill_val = y_raw.mode()[0] if not y_raw.mode().empty else 0
            self.y = self.label_encoder.fit_transform(y_raw.fillna(fill_val))
        else:
            self.y = y_raw.fillna(y_raw.mean()).values

    def tune_and_train(self, n_trials: int = 5, epochs: int = 5):
        """Run ultra-fast hyperparameter optimization (5 trials, 5 epochs)."""
        print(f"[trainer] 🧪 Starting hyper-tuning ({n_trials} trials, optimized for demo speed)...")
        
        self.study = optuna.create_study(direction="minimize" if self.task_type == "regression" else "maximize")
        
        def objective(trial):
            if any(x in self.architecture for x in ["xgboost", "random forest", "auto"]):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 100),
                    "max_depth": trial.suggest_int("max_depth", 3, 7),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                }
                
                if self.task_type == "classification":
                    model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
                    score = cross_val_score(model, self.X, self.y, cv=2, scoring='accuracy').mean()
                else:
                    model = xgb.XGBRegressor(**params, random_state=42)
                    score = -cross_val_score(model, self.X, self.y, cv=2, scoring='neg_mean_squared_error').mean()
                return score
            
            else:
                # Optimized PyTorch MLP study
                input_dim = self.X.shape[1]
                hidden_dim = trial.suggest_int("hidden_dim", 32, 64)
                lr = trial.suggest_float("lr", 1e-3, 5e-2, log=True)
                
                if self.task_type == "classification":
                    output_dim = len(np.unique(self.y))
                    model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))
                    criterion = nn.CrossEntropyLoss()
                else:
                    model = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
                    criterion = nn.MSELoss()
                
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                X_t, y_t = torch.tensor(self.X, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long if self.task_type == "classification" else torch.float32)
                if self.task_type == "regression": y_t = y_t.view(-1, 1)

                # 2 epochs proxy for trials
                for _ in range(2):
                    optimizer.zero_grad(); loss = criterion(model(X_t), y_t); loss.backward(); optimizer.step()
                return loss.item()

        self.study.optimize(objective, n_trials=n_trials)
        self.best_params = self.study.best_params
        print(f"[trainer] 🏆 Best Trial Params: {self.best_params}")
        
        # ── Final Fit for Judges (Fast 5 - 10 Epochs) ──────────────────────────
        if any(x in self.architecture for x in ["xgboost", "random forest", "auto"]):
            if self.task_type == "classification":
                self.model = xgb.XGBClassifier(**self.best_params, random_state=42)
            else:
                self.model = xgb.XGBRegressor(**self.best_params, random_state=42)
            self.model.fit(self.X, self.y)
        else:
            # Final PyTorch fit
            input_dim = self.X.shape[1]
            self.model = nn.Sequential(nn.Linear(input_dim, self.best_params.get("hidden_dim", 64)), nn.ReLU(), 
                                       nn.Linear(self.best_params.get("hidden_dim", 64), len(np.unique(self.y)) if self.task_type == "classification" else 1))
            criterion = nn.CrossEntropyLoss() if self.task_type == "classification" else nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.best_params.get("lr", 1e-3))
            X_t, y_t = torch.tensor(self.X, dtype=torch.float32), torch.tensor(self.y, dtype=torch.long if self.task_type == "classification" else torch.float32)
            if self.task_type == "regression": y_t = y_t.view(-1, 1)

            for epoch in range(epochs):
                optimizer.zero_grad(); loss = criterion(self.model(X_t), y_t); loss.backward(); optimizer.step()
                if epoch % 1 == 0: print(f"[trainer] Judge Fit Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}")

        self._save_artifacts()

    def _save_artifacts(self):
        """Save results, metrics and plots for the OmniML final report."""
        from sklearn.metrics import confusion_matrix
        metrics = {"task": self.task_type, "architecture": self.architecture, "best_params": self.best_params, "eval_score": self.study.best_value, "samples": len(self.X)}
        with open("metrics.json", "w") as f: json.dump(metrics, f, indent=4)
            
        plt.figure(figsize=(8, 5)); trials = [t.number for t in self.study.trials]; values = [t.value for t in self.study.trials]
        plt.plot(trials, values, marker='o', linestyle='-', color='#636EFA'); plt.title("OmniML Optimization History"); plt.xlabel("Trial"); plt.ylabel("Score"); plt.grid(True); plt.savefig("loss_curve.png"); plt.close()
        
        plt.figure(figsize=(8, 6)); corr = pd.DataFrame(self.X[:, :10], columns=self.scaler.active_columns[:10]).corr()
        sns.heatmap(corr, annot=True, cmap="mako", cbar=False); plt.title("Industrial Feature Correlation (Top 10)"); plt.savefig("telemetry_distribution.png"); plt.close()
        
        # --- NEW: Binary Classification Confusion Matrix ---
        if self.task_type == "classification":
            X_torch = torch.tensor(self.X, dtype=torch.float32)
            if hasattr(self.model, "predict"):
                y_pred = self.model.predict(self.X)
            else:
                with torch.no_grad():
                    logits = self.model(X_torch)
                    y_pred = torch.argmax(logits, dim=1).numpy()
            
            cm = confusion_matrix(self.y, y_pred)
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title("Classification Confusion Matrix")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.tight_layout()
            plt.savefig("confusion_matrix.png")
            plt.close()
            print("[trainer] 📊 Confusion matrix exported successfully.")

        print("[trainer] ✅ All judge-ready artifacts exported.")


