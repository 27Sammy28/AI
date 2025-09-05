# train_batch.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from models.lstm_baseline import LSTMForecast

def create_sequences(X, y, seq_len=8):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.stack(Xs), np.stack(ys)

def train(path, feature_cols, label_col, seq_len=8, epochs=10, batch_size=64, lr=1e-3):
    df = pd.read_csv(path).dropna(subset=feature_cols+[label_col])
    X = df[feature_cols].values.astype(np.float32)
    y = df[label_col].values.astype(np.float32)
    Xs, ys = create_sequences(X, y, seq_len)
    ds = TensorDataset(torch.from_numpy(Xs), torch.from_numpy(ys))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = LSTMForecast(input_dim=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for ep in range(epochs):
        tot = 0
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * xb.size(0)
        print(f"Epoch {ep+1}/{epochs}, loss={tot/len(ds):.6f}")
    torch.save(model.state_dict(), "models/lstm_baseline.pth")
    return model

