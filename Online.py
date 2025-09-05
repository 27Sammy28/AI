# online/ogd.py
import numpy as np

class OnlineGD:
    """
    Online Gradient Descent for linear predictors: y_hat = w^T x
    Supports squared loss (regression) and logistic (classification).
    """
    def __init__(self, dim, lr=0.01, projection_radius=None, loss="squared"):
        self.w = np.zeros(dim)
        self.lr = lr
        self.projection_radius = projection_radius  # if None, no projection
        assert loss in ("squared", "logistic")
        self.loss = loss

    def predict(self, x):
        return float(np.dot(self.w, x))

    def _gradient(self, x, y):
        if self.loss == "squared":
            y_hat = np.dot(self.w, x)
            grad = 2.0 * (y_hat - y) * x
            return grad
        elif self.loss == "logistic":
            # logistic binary: y in {0,1}
            z = np.dot(self.w, x)
            s = 1 / (1 + np.exp(-z))
            grad = (s - y) * x
            return grad

    def _project(self):
        if self.projection_radius is None:
            return
        norm = np.linalg.norm(self.w)
        if norm > self.projection_radius:
            self.w = (self.projection_radius / norm) * self.w

    def update(self, x, y, t=None):
        grad = self._gradient(x, y)
        eta = self.lr if t is None else self.lr / np.sqrt(max(1, t))
        self.w = self.w - eta * grad
        self._project()

