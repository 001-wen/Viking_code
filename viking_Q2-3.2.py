import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 3 * X + np.random.randn(100, 1)+4

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        self.loss_history = []

    def fit(self, X, y):
        m = len(X)
        self.w = np.random.randn()
        self.b = np.random.randn()
        for i in range(self.n_iterations):
            y_pred = X * self.w + self.b
            loss = (1 / m) * np.sum((y_pred - y) ** 2)
            self.loss_history.append(loss)
            dw = 2 * (np.sum(X * (y_pred - y))) / m
            db = 2 * (np.sum(y_pred - y)) / m
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db

    def predict(self, X):
        return X * self.w + self.b

# 学习率与损失函数的关系
def plot_loss_iterations(n_iterations, learning_rates):
    fig1 = plt.figure(figsize=(10,7))
    for lr in learning_rates:
        model = LinearRegression(learning_rate=lr, n_iterations=n_iterations)
        model.fit(X, y)
        plt.plot(model.loss_history, label=f'α={lr}')
    plt.title('Loss-Learn-rate')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()

learning_rates = [0.01, 0.05, 0.1]
n_iterations = 1000
plot_loss_iterations(n_iterations, learning_rates)

# 损失与迭代次数的关系
def plot_loss_learning_rate(learning_rate, n_iterations):
    fig2 = plt.figure(figsize=(10,7))
    model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
    model.fit(X, y)
    plt.plot(model.loss_history)
    plt.title(f'Loss-(α={learning_rate})')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.grid()

plot_loss_learning_rate(0.01, 1000)

# 拟合曲线与样本的关系
def plot_x_y(X, y):
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    fig_3 = plt.figure(figsize=(10,7))
    plt.scatter(X, y, color='green')
    plt.plot(X, y_pred, color='red')
    plt.title('Linear Regression Fit')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.grid()

plot_x_y(X,y)
plt.show()