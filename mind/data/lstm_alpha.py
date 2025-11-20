import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from typing import Optional
from .alpha import AlphaModel


class LSTMNet(nn.Module):
    """
    A simple LSTM network for predicting asset returns.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1
    ):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(
            x, (h0, c0)
        )  # out: tensor of shape (batch_size, seq_length, hidden_dim)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class LSTMAlphaModel(AlphaModel):
    """
    Alpha model that uses a pre-trained LSTM to predict expected returns.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        input_dim: int = 1,
        sequence_length: int = 10,
    ):
        """
        Args:
            model_path: Path to load the pre-trained model state dict. If None, initializes random weights.
            input_dim: Number of features per time step (e.g., 1 for just Close price).
            sequence_length: Number of past time steps to look at.
        """
        self.sequence_length = sequence_length
        self.input_dim = input_dim

        # Initialize model architecture
        # Assuming 1 output per asset (regression for expected return)
        # Note: In a real scenario, you might have one model per asset or a global model.
        # Here we assume a global model applied to each asset independently for simplicity.
        self.model = LSTMNet(input_dim=input_dim, hidden_dim=32, output_dim=1)

        if model_path:
            try:
                self.model.load_state_dict(torch.load(model_path))
                self.model.eval()
            except FileNotFoundError:
                print(
                    f"Warning: Model file {model_path} not found. Using random weights."
                )

    def predict(self, market_history: pd.DataFrame) -> pd.Series:
        """
        Predict expected returns using the LSTM model.

        Args:
            market_history: DataFrame with market data.

        Returns:
            pd.Series: Predicted returns for each asset.
        """
        # Pivot to get Close prices: Index=Date, Columns=Ticker
        df = market_history.copy()
        if "ticker" in df.columns and "close" in df.columns:
            df = df.pivot(index="timestamp", columns="ticker", values="close")

        predictions = {}

        # Iterate over each asset to make a prediction
        # This is a simplification. A more advanced model might process all assets in a batch.
        for ticker in df.columns:
            series = df[ticker].dropna()

            if len(series) < self.sequence_length:
                # Not enough data, return 0 neutral alpha
                predictions[ticker] = 0.0
                continue

            # Prepare input sequence
            # Taking the last 'sequence_length' points
            seq = series.iloc[-self.sequence_length :].values

            # Normalize (simple pct change or standardization is better, but using raw/scaled for example)
            # Here we just use raw values for the structure, but in practice you MUST normalize.
            # Let's assume the model expects normalized returns.
            seq_returns = np.diff(seq) / seq[:-1]
            # Pad the one lost value from diff or just take last sequence_length returns

            # Re-fetching sequence of returns directly might be better
            # Let's calculate returns for the whole series first
            returns = series.pct_change().dropna()
            if len(returns) < self.sequence_length:
                predictions[ticker] = 0.0
                continue

            seq_input = returns.iloc[-self.sequence_length :].values

            # Convert to tensor: (Batch=1, Seq, InputDim)
            x = torch.tensor(seq_input, dtype=torch.float32).view(
                1, self.sequence_length, 1
            )

            # Inference
            with torch.no_grad():
                out = self.model(x)
                predicted_return = out.item()

            predictions[ticker] = predicted_return

        return pd.Series(predictions)
