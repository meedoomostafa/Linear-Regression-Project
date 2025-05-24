import gradient_decent as gd
import regression_utilities as ru
import numpy as np
from sklearn.metrics import r2_score


class LinearRegression:
    def __init__(self):
        self.weights = None
        self._means = None
        self._stds = None

    def fit(self, X, y):
        # Normalize features and store parameters
        X_normalized, self._means, self._stds = ru.normalize_features(X)
        X_with_bias = ru.add_bias(X_normalized)

        # Initialize weights
        self.weights = np.zeros(X_with_bias.shape[1])

        # Gradient Descent
        step_sizes = [0.1, 0.01, 0.001]
        precisions = [0.1, 0.01, 0.001]
        self.weights, _ = gd.gradient_decent(
            X_with_bias, y, self.weights, step_sizes, precisions
        )

    def predict(self, X):
        # Normalize using training parameters
        X_normalized = (X - self._means) / self._stds
        X_with_bias = ru.add_bias(X_normalized)
        return X_with_bias @ self.weights

    def accuracy(self, y_true, X):
        y_pred = self.predict(X)
        return r2_score(y_true, y_pred)*100
    def plot_regression_line(self, X, y):
        import matplotlib.pyplot as plt
        plt.scatter(X, y, color='blue')
        plt.plot(X, self.predict(X), color='red')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title('Regression Line')
        plt.show()
        
    def plot_cost_function(self):
        import matplotlib.pyplot as plt
        plt.plot(self.cost_history)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function')
        plt.show()