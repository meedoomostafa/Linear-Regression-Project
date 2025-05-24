import data_helper as dh
from model import LinearRegression
import numpy as np

print('Loading data...')
data = dh.load_data('dataset_200x4_regression.csv')
x = data['x']
y = data['y']

print('Processing data...')
model = LinearRegression()
model.fit(x, y)
print('Data processed!')

predictions = model.predict(x)

print('Weights:', model.weights)
print('First 5 predictions:', predictions[:5])
print('Accuracy (R²):', model.accuracy(y, x))  # Pass both y and X

# Optional: plot the regression line if feature is 1-dimensional
if len(x.shape) == 1 or (len(x.shape) > 1 and x.shape[1] == 1):
    model.plot_regression_line(x, y)

