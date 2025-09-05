# utils/regret.py
import numpy as np
from sklearn.linear_model import Ridge
from typing import List

def empirical_regret_squared(X: np.ndarray, y: np.ndarray, online_losses: List[float]) -> float:
    """
    For squared loss, compute best fixed w* by solving min_w sum_t (w^T x_t - y_t)^2.
    We use Ridge (alpha small) to stabilize.
    Returns R_emp = sum_online_losses - sum_offline_losses
    """
    # fit offline linear regressor
    model = Ridge(alpha=1e-6, fit_intercept=False)
    model.fit(X, y)
    y_pred = model.predict(X)
    offline_losses = np.sum((y_pred - y)**2)
    online_sum = np.sum(online_losses)
    return online_sum - offline_losses, offline_losses

# if you want vectorized losses per timestep, return arrays for plotting
def per_timestep_offline_losses(X, y, model):
    y_pred = model.predict(X)
    return (y_pred - y)**2

