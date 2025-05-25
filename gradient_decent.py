import numpy as np
from numpy.linalg import norm


def gradient_decent(features, targets, initial_weights, step_sizes, precisions, iterations=100000):
    best_weights = None
    min_error = float('inf')
    n_samples = features.shape[0]

    for step in step_sizes:
        for precision in precisions:
            weights = initial_weights.copy()
            for i in range(iterations):
                # Compute gradient for MSE
                predictions = features @ weights
                errors = predictions - targets
                gradient = (features.T @ errors) / n_samples

                # Update weights
                new_weights = weights - step * gradient

                # Check convergence
                if norm(new_weights - weights) < precision:
                    break
                weights = new_weights

            # Evaluate with MSE
            current_error = np.mean(errors ** 2)
            if current_error < min_error:
                min_error = current_error
                best_weights = weights.copy()

    return best_weights, []