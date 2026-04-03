import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, List

class AutoScaler:
    """
    AutoScaler utility for dataset-blind ingestion.
    Silently drops constants and remembers the exact dimensionality of active telemetry.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.active_columns = []

    def detect_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates variance across numeric columns and drops those with Variance < 1e-6.
        """
        # Ensure numerics only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate var and filter
        variances = numeric_df.var()
        self.active_columns = variances[variances >= 1e-6].index.tolist()
        
        return df[self.active_columns]

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Detects active features, fits the scaler, and returns scaled array."""
        filtered_df = self.detect_features(df)
        return self.scaler.fit_transform(filtered_df)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms incoming dataframe using the previously detected features."""
        if not self.active_columns:
            raise RuntimeError("AutoScaler hasn't been fitted. Active columns empty.")
        # Only extract the learned active columns natively, ignoring everything else
        return self.scaler.transform(df[self.active_columns])


class DynamicLSTMAutoencoder(nn.Module):
    """
    Dynamic LSTM Autoencoder for feature-agnostic reconstruction-based anomaly detection.
    """
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        """
        Initialize the Autoencoder with dynamic input dimensions.
        """
        super(DynamicLSTMAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.encoder = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                               num_layers=num_layers, batch_first=True)
        self.decoder = nn.LSTM(input_size=hidden_size, hidden_size=input_size, 
                               num_layers=num_layers, batch_first=True)
        # Threshold for alerting, fit via train_model
        self.anomaly_threshold = None

    def reinitialize_weights(self, new_input_size: int):
        """
        Safely destroys and rebuilds the recurrent blocks if shape dynamically jumps between fit() runs.
        """
        if self.input_size != new_input_size:
            self.input_size = new_input_size
            self.encoder = nn.LSTM(input_size=new_input_size, hidden_size=self.hidden_size, 
                                   num_layers=self.num_layers, batch_first=True)
            self.decoder = nn.LSTM(input_size=self.hidden_size, hidden_size=new_input_size, 
                                   num_layers=self.num_layers, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for reconstruction.
        """
        # Encode
        _, (hidden_state, _) = self.encoder(x)
        
        # Decode - we repeat the last hidden state for the sequence length
        seq_len = x.shape[1]
        
        # Use the hidden state from the last layer (idx -1)
        hidden_last = hidden_state[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        decoded, _ = self.decoder(hidden_last)
        return decoded

    def train_model(self, scaled_data: np.ndarray, epochs: int = 50, batch_size: int = 32, lr: float = 0.001) -> float:
        """
        Trains the autoencoder and calculates the 95th percentile MSE for dynamic thresholding.
        """
        self.train()
        dataset = torch.tensor(scaled_data, dtype=torch.float32)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            for batch in dataloader:
                optimizer.zero_grad()
                reconstruction = self(batch)
                loss = criterion(reconstruction, batch)
                loss.backward()
                optimizer.step()

        # Compute dynamic threshold strictly from training MSEs
        self.eval()
        with torch.no_grad():
            train_reconstruction = self(dataset)
            mse_scores = torch.mean((dataset - train_reconstruction) ** 2, dim=(1, 2)).numpy()
            
        # 95th percentile sets the threshold
        self.anomaly_threshold = float(np.percentile(mse_scores, 95))
        return self.anomaly_threshold

    def calculate_mse(self, scaled_live_data: np.ndarray) -> np.ndarray:
        """
        Calculates reconstruction error for incoming data.
        """
        self.eval()
        tensor_data = torch.tensor(scaled_live_data, dtype=torch.float32)
        with torch.no_grad():
            reconstruction = self(tensor_data)
        
        mse = torch.mean((tensor_data - reconstruction) ** 2, dim=(1, 2)).numpy()
        return mse
