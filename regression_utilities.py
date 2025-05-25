import numpy as np

def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def normalize_features(X):
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1  # Avoid division by zero
    X_normalized = (X - means) / stds
    return X_normalized, means, stds

def cost_function(X, y, weights):
    m = len(y)
    errors = y - (X @ weights)
    return (errors.T @ errors) / (2 * m)