import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    m,n = X.shape
    w = np.zeros(n, dtype=float)
    b = 0.0
    for _ in range(int(steps)):
        z = X@w + b
        a = _sigmoid(np.clip(z, -500, 500))
        diff = a - y

        dw = (X.T @ diff) / m
        db = np.sum(diff) / m   

        w -= lr * dw
        b -= lr * db
    return w,b
    pass