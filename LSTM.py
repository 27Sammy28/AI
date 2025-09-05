# models/lstm_baseline.py
import torch
import torch.nn as nn

class LSTMForecast(nn.Module):
    """
    Simple LSTM regressor:
    Input: (batch, seq_len, feature_dim)
    We'll train offline on sequences, but use it as baseline.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        # x: [B, T, F]
        out, (hn, cn) = self.lstm(x)
        # take last timestep
        h_last = out[:, -1, :]
        return self.fc(h_last).squeeze(-1)

