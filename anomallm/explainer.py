import pandas as pd
import numpy as np
import networkx as nx
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from typing import Tuple, List, Optional

class GrangerExplainer:
    """
    Causality Engine using VAR Model and Granger Causality tests to compute root cause and cascade analysis.
    """
    def __init__(self, maxlags: int = 3, significance_level: float = 0.05):
        """
        Initialize the Granger Explainer.
        
        Args:
            maxlags (int): The strict constraint for the maximum number of lags for VAR and Granger causality.
            significance_level (float): The p-value threshold to accept causality.
        """
        self.maxlags = maxlags
        self.significance_level = significance_level

    def _make_stationary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies differencing if required to make time-series columns stationary.
        """
        stationary_df = df.copy()
        for col in stationary_df.columns:
            # Check stationarity using Augmented Dickey-Fuller
            try:
                result = adfuller(stationary_df[col].dropna())
                if result[1] > self.significance_level:
                    # Need differencing
                    stationary_df[col] = stationary_df[col].diff().fillna(0)
            except Exception:
                # If ADF fails (e.g., constant values), fill with 0
                stationary_df[col] = 0
        return stationary_df

    def analyze(self, anomaly_window_df: pd.DataFrame) -> Tuple[str, List[str]]:
        """
        Execute Granger Causality across the isolated anomalous dataframe window.
        
        Args:
            anomaly_window_df (pd.DataFrame): Dataframe of the specific anomaly time window.
            
        Returns:
            Tuple[str, List[str]]: The singular root cause column and an ordered list representing the cascade path.
        """
        # Ensure stationarity
        df = self._make_stationary(anomaly_window_df)
        
        # Remove columns with zero variance (no information content)
        variances = df.var()
        valid_cols = variances[variances > 1e-10].index.tolist()
        
        if len(valid_cols) < 2:
            root_cause = valid_cols[0] if valid_cols else "Unknown"
            return root_cause, []

        df = df[valid_cols]
        
        # Train VAR Model
        try:
            model = VAR(df)
            
            # Select best lag within constraints constraint
            max_possible_lags = min(self.maxlags, (len(df) // (2 * len(valid_cols))) - 1)
            
            if max_possible_lags < 1:
                # If the window size is too small for VAR given the feature count, return basic stats
                highest_var_col = anomaly_window_df[valid_cols].var().idxmax()
                return highest_var_col, [highest_var_col]
                
            res = model.fit(maxlags=max_possible_lags, ic='aic')
            optimal_lag = res.k_ar
            
            if optimal_lag == 0:
                highest_var_col = anomaly_window_df[valid_cols].var().idxmax()
                return highest_var_col, [highest_var_col]

            # Test causality between all pairs
            G = nx.DiGraph()
            G.add_nodes_from(valid_cols)
            
            for caused in valid_cols:
                for causer in valid_cols:
                    if caused != causer:
                        # test_causality returns a dictionary with different tests
                        try:
                            test_result = res.test_causality(caused, causer, kind='f')
                            p_value = test_result.pvalue
                            if p_value < self.significance_level:
                                G.add_edge(causer, caused, weight=1.0 - p_value)
                        except Exception:
                            pass

            if len(G.edges) == 0:
                highest_var_col = anomaly_window_df[valid_cols].var().idxmax()
                return highest_var_col, []

            # Calculate out-degree for root cause (who causes the most other variables)
            out_degrees = dict(G.out_degree())
            root_cause = max(out_degrees, key=out_degrees.get)
            
            # Cascade path: traversing from the root cause based on causality graph edges
            cascade_path = [root_cause]
            current_node = root_cause
            visited = set([root_cause])
            
            while True:
                edges = G.out_edges(current_node, data=True)
                if not edges:
                    break
                    
                # Pick strongest causal link pointing outward
                next_node = sorted(edges, key=lambda x: x[2]['weight'], reverse=True)[0][1]
                if next_node in visited:
                    break
                cascade_path.append(next_node)
                visited.add(next_node)
                current_node = next_node

            return root_cause, cascade_path
            
        except Exception:
            # Fallback in case VAR model fails to converge
            highest_var_col = anomaly_window_df[valid_cols].var().idxmax()
            return highest_var_col, [highest_var_col]
