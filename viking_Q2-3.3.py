import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

california = fetch_california_housing()
X = california.data
y = california.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def least_squares(X, y):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_best

theta = least_squares(X_train, y_train)
X_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
y_predict = X_test_b.dot(theta)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_predict)
plt.plot([y.min(), y.max()], [y.min(), y.max()], c='red', lw=2)
plt.xlabel('reality')
plt.ylabel('prediction')
plt.title('viking')
plt.grid()
plt.show()