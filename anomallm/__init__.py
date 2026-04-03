import os
# Hardware-agnostic thread suppression (Prevents Mac/Linux OpenMP segmentation faults)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import numpy as np
import pandas as pd
import re
from typing import Dict, Any, Tuple, Optional

from .detector import DynamicLSTMAutoencoder, AutoScaler
from .explainer import GrangerExplainer
from .reporter import LLMReporter

__all__ = ["AnomaLLM"]

class AnomaLLM:
    """
    The main AnomaLLM class that exposes the end-to-end pipeline for multivariate anomaly detection.
    Dataset-blind edition.
    """
    def __init__(self, window_size: int = 30, llm_url: str = "http://localhost:11434/v1", llm_model: str = "llama3", max_lags: int = 3):
        """
        Initializes the AnomaLLM SDK pipeline.
        """
        self.window_size = window_size
        
        # Internal modules
        self.scaler = AutoScaler()
        self.detector = None
        self.explainer = GrangerExplainer(maxlags=max_lags)
        self.reporter = LLMReporter(base_url=llm_url, model=llm_model)
        
        # Extracted Metadata state
        self.metadata_cols = []
        self.id_col = None

    def _discover_metadata(self, df: pd.DataFrame):
        """
        Regex-based discovery to identify columns representing IDs or timestamps.
        """
        self.metadata_cols = []
        self.id_col = None
        
        keywords = r'(time|date|id|sn|serial|timestamp)'
        
        for col in df.columns:
            # Check string dtypes
            if not pd.api.types.is_numeric_dtype(df[col]):
                self.metadata_cols.append(col)
                continue
                
            # Check for regex matches in column name for numerical identifiers
            if re.search(keywords, str(col).lower()):
                self.metadata_cols.append(col)
                if 'id' in str(col).lower() or 'sn' in str(col).lower():
                    self.id_col = col

    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        """
        Slides a window across the array to create 3D sequence arrays.
        """
        sequences = []
        for i in range(len(data) - self.window_size + 1):
            sequences.append(data[i:i + self.window_size])
        return np.array(sequences)

    def fit(self, df: pd.DataFrame) -> None:
        """
        Dataset-blind ingestion. Strips metadata, detects valid features, and trains the LSTM dynamically.
        """
        # 1. Discover and isolate metadata
        self._discover_metadata(df)
        
        # 2. Extract pure numeric feature matrix (ignoring metadata columns)
        feature_cols = [c for c in df.columns if c not in self.metadata_cols and pd.api.types.is_numeric_dtype(df[c])]
        working_df = df[feature_cols].copy()
        
        # 3. AutoScaler drops constants (variance < 1e-6) and fits Standardizer
        scaled_data = self.scaler.fit_transform(working_df)
        
        self.n_features = len(self.scaler.active_columns)
        if self.n_features == 0:
            raise ValueError("No active telemetry sensors discovered. All columns were constant or metadata.")
        
        # 4. Convert to 3D sequences
        sequences = self._create_sequences(scaled_data)
        
        # 5. Dynamic Model Initialization
        if self.detector is None:
            self.detector = DynamicLSTMAutoencoder(input_size=self.n_features, hidden_size=max(16, self.n_features * 2))
        else:
            self.detector.reinitialize_weights(self.n_features)
            
        # 6. Train the model
        self.detector.train_model(sequences)

    def detect(self, df_live: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Scans live data, automatically coercing matching columns and checking thresholds.
        """
        if self.detector is None:
            raise RuntimeError("Model hasn't been fitted. Call fit() before detect().")

        # 1. Ensure required columns are present before extraction
        missing = set(self.scaler.active_columns) - set(df_live.columns)
        if missing:
            raise KeyError(f"Live data missing active telemetry columns: {missing}")
            
        # 2. Extract matching features and scale dynamically
        scaled_live = self.scaler.transform(df_live)
        
        if len(scaled_live) < self.window_size:
            return None # Window too short
            
        # 3. Sequencing
        sequences = self._create_sequences(scaled_live)
        
        # 4. Error mapping
        mse_scores = self.detector.calculate_mse(sequences)
        max_idx = int(np.argmax(mse_scores))
        max_mse = float(mse_scores[max_idx])
        
        if max_mse > self.detector.anomaly_threshold:
            # Recreate raw anomaly window including metadata so explain() can use it
            anomaly_window_raw = df_live.iloc[max_idx : max_idx + self.window_size].copy()
            anomaly_window_raw.attrs['anomaly_score'] = max_mse
            return anomaly_window_raw

        return None

    def explain(self, anomaly_window_df: pd.DataFrame, asset_id: str = None, mode: str = "Technical Diagnostics") -> Dict[str, Any]:
        """
        Executes causal explainer against the window and generates LLM insight report.
        """
        score = anomaly_window_df.attrs.get('anomaly_score', -1.0)
        
        # Dataset-blind Asset ID resolution
        resolved_asset_id = "Asset_001"
        if asset_id:
            resolved_asset_id = asset_id
        elif self.id_col and self.id_col in anomaly_window_df.columns:
            # Grab the ID from the most recent row of the incident window
            extracted_id = anomaly_window_df[self.id_col].iloc[-1]
            resolved_asset_id = f"{self.id_col}_{extracted_id}"
            
        # Causality engine only operates on the active telemetry columns
        active_window = anomaly_window_df[self.scaler.active_columns]
        root_cause, cascade_path = self.explainer.analyze(active_window)
        
        llm_report = self.reporter.generate_incident_report(
            asset_id=resolved_asset_id,
            mse_score=score,
            root_cause=root_cause,
            cascade_path=cascade_path,
            mode=mode,
            feature_names=self.scaler.active_columns
        )
        
        return {
            "root_cause": root_cause,
            "cascade": cascade_path,
            "llm_report": llm_report,
            "mse_score": score,
            "asset_id_resolved": resolved_asset_id
        }
