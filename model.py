import gradient_decent as gd
import regression_utilities as ru
import numpy as np
from sklearn.preprocessing import MinMaxScaler


class LinearRegression:
    def __init__(self):
        self.weights = None
        self.cost = None
        self.states_list = None
        self.predictions = None
        self.errors = None

    def fit(self, X, y):
        step_size = [0.1,0.01,0.001,0.0001,0.00001]
        precision = [0.1,0.01,0.001,0.0001,0.00001]

        X = ru.add_bias(X)
        self.weights = np.zeros(X.shape[1])

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

        result = gd.gradient_decent(X, y, self.weights, step_size, precision, 1000)
        self.weights = result[0]
        self.states_list = result[1]
        self.cost = ru.cost_function(X, y, self.weights)
        self.predictions = ru.predictions(X, self.weights)
        self.errors = gd.errors(y, self.predictions)

    def predict(self, X):
        X = ru.add_bias(X)
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
        return ru.predictions(X, self.weights)

    def plot_cost_function(self):
        ru.plot_cost_function(self.errors, self.predictions)

    def plot_regression_line(self, X, y):
        ru.plot_regression_line(X, y, self.predictions)

    def accuracy(self, y):
        return (np.sum(np.abs(self.errors)) / np.sum(np.abs(y))) * 100