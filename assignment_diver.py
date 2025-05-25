import data_helper as dh
from model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print('Loading data...')
data = dh.load_data('dataset_200x4_regression.csv')
X = data['x']
y = data['y']


print('Splitting data...')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Processing data...')
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_poly, y_train)
print('Data processed!')

train_predictions = model.predict(X_train_poly)
test_predictions = model.predict(X_test_poly)

print('Weights:', model.weights)

print('First 5 Training Predictions:', train_predictions[:5])
print('First 5 Test Predictions:', test_predictions[:5])

train_accuracy = model.accuracy(y_train, X_train_poly)
test_accuracy = model.accuracy(y_test, X_test_poly)
train_mse = mean_squared_error(y_train, train_predictions)
test_mse = mean_squared_error(y_test, test_predictions)

print('Training Accuracy:', train_accuracy, '%')
print('Test Accuracy:', test_accuracy, '%')
print('Training MSE:', train_mse)
print('Test MSE:', test_mse)

from sklearn.decomposition import PCA
pca = PCA(n_components=1)
X_train_reduced = pca.fit_transform(X_train_poly)
X_test_reduced = pca.transform(X_test_poly)

model.plot_regression_line(X_train_reduced, y_train)
model.plot_regression_line(X_test_reduced, y_test)
