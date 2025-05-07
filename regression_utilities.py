import numpy as np
import matplotlib.pyplot as plt

# cost function
def cost_function(y , y_pred):
    total = sum ((y - y_pred) ** 2)
    return total

def plot_cost_function(y, y_pred):
    plt.plot(y, y_pred)
    plt.xlabel('y')
    plt.ylabel('y_pred')
    plt.title('Cost Function')
    plt.show()


def plot_regression_line(x, y, y_pred):
    plt.scatter(x, y, color='blue', label='Data points')
    plt.plot(x, y_pred, color='red', label='Regression line')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend()
    plt.show()

# adding bias
def add_bias(x):
    return np.c_[np.ones((x.shape[0], 1)), x]