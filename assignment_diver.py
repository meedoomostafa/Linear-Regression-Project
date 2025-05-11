import data_helper as dh
from model import LinearRegression

data = dh.load_data('dataset_200x4_regression.csv')
x = data['x']
y = data['y']

model = LinearRegression()
print('processing data ...')
model.fit(x, y)
print('data processed ...')
predictions = model.predict(x)

weight = model.weights

print('Weights:', weight)
print('First 5 predictions:', predictions[:5])
print('Accuracy (R²):', model.accuracy(y))


