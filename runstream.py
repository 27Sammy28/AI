# run_stream.py
import numpy as np
import matplotlib.pyplot as plt
from utils.data_loader import stream_csv
from online.ogd import OnlineGD
from utils.regret import empirical_regret_squared
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd

def run_stream_experiment(csv_path, feature_cols, label_col, max_steps=None, lr=1e-3, projection_radius=None):
    # load all data for offline regret computation later
    df = pd.read_csv(csv_path).dropna(subset=feature_cols+[label_col])
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
    X_all = df[feature_cols].values.astype(float)
    y_all = df[label_col].values.astype(float)

    # scaling using training mean (simple)
    scaler = StandardScaler()
    X_all = scaler.fit_transform(X_all)

    # streaming generator uses scaled X
    def stream_generator():
        for xi, yi in zip(X_all, y_all):
            yield xi, yi

    dim = X_all.shape[1]
    online = OnlineGD(dim=dim, lr=lr, projection_radius=projection_radius, loss="squared")
    online_losses = []
    w_history = []
    preds = []
    ys = []

    gen = stream_generator()
    steps = 0
    for x, y in gen:
        steps += 1
        y_hat = online.predict(x)
        loss_t = (y_hat - y)**2
        online.update(x, y, t=steps)
        online_losses.append(loss_t)
        w_history.append(online.w.copy())
        preds.append(y_hat)
        ys.append(y)
        if max_steps and steps >= max_steps:
            break

    # compute regret vs offline best (using all seen samples)
    X_seen = np.array(list(X_all[:len(ys)]))
    y_seen = np.array(list(ys))
    R_empirical, offline_losses = empirical_regret_squared(X_seen, y_seen, online_losses)
    print(f"Empirical regret (sum online - offline): {R_empirical:.4f}")
    # plot loss curve
    plt.figure(figsize=(8,4))
    plt.plot(np.cumsum(online_losses), label="Cumulative online loss")
    # plot cumulative offline loss baseline for reference
    offline_cum = np.cumsum((np.linalg.pinv(X_seen.T @ X_seen) @ X_seen.T @ y_seen - y_seen)**2) if False else None
    plt.title("Cumulative online loss")
    plt.xlabel("t")
    plt.ylabel("cumulative loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/cumulative_online_loss.png", dpi=150)
    return {
        "R_empirical": R_empirical,
        "online_losses": online_losses,
        "preds": preds,
        "ys": ys,
    }

if __name__ == "__main__":
    csv_path = "data/ace_dscovr.csv"  # put your CSV here
    features = ["Bx", "By", "Bz", "speed", "density"]  # adapt to your file
    label = "speed"
    res = run_stream_experiment(csv_path, features, label, max_steps=2000, lr=1e-3, projection_radius=10.0)
    print("Done.")

