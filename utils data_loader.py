# utils/data_loader.py
import pandas as pd
import numpy as np
from typing import Iterator, Tuple

def stream_csv(path: str, feature_cols: list, label_col: str, dropna=True) -> Iterator[Tuple[np.ndarray, float]]:
    """
    Yield one sample at a time (x_t, y_t) from a timestamped CSV.
    CSV must contain rows ordered by time (oldest -> newest).

    Example usage:
      for x, y in stream_csv("data/ace.csv", features, "solar_wind_speed"):
          ...
    """
    df = pd.read_csv(path)
    if dropna:
        df = df.dropna(subset=feature_cols + [label_col])
    # Optionally: sort by time column if present
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    for _, row in df.iterrows():
        x = row[feature_cols].astype(float).to_numpy()
        y = float(row[label_col])
        yield x, y

