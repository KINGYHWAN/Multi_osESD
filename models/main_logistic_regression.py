import numpy as np


class LogisticRegressionSequential:
    def __init__(self):
        self.w = None
        self.b = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, dim):
        self.w = np.zeros((dim, 1))
        self.b = 0

    def forward_propagation(self, X):
        z = np.dot(X, self.w) + self.b
        a = self.sigmoid(z)
        return a

    def compute_cost(self, a, y):
        m = len(y)
        cost = (-1/m) * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a))
        return cost

    def backward_propagation(self, X, a, y):
        m = len(y)
        dz = a - y
        dw = (1/m) * np.dot(X.T, dz)
        db = (1/m) * np.sum(dz)
        return dw, db

    def update_parameters(self, lrs, dw, db):
        self.w = self.w - lrs[:-1] * dw
        self.b = self.b - lrs[-1] * db

    def train(self, X, y, num_iterations, lrs):
        m, n = X.shape
        self.initialize_parameters(n)
        for i in range(num_iterations):
            a = self.forward_propagation(X)
            cost = self.compute_cost(a, y)
            dw, db = self.backward_propagation(X, a, y)
            self.update_parameters(lrs, dw, db)

    def train_incremental(self, X_new, y_new, lrs):
        a = self.forward_propagation(X_new)
        cost = self.compute_cost(a, y_new)
        dw, db = self.backward_propagation(X_new, a, y_new)
        self.update_parameters(lrs, dw, db)
    
    def predict(self, X):
        a = self.forward_propagation(X)
        predictions = (a >= 0.5).astype(int)
        return predictions




