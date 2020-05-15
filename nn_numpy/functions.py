import numpy as np

# %%
class Linear():
    def __init__(self, n_input, n_output):
        self.weights = np.random.randn(n_input, n_output)
        self.bias = np.zeros(n_output)

    def forward(self, x):
        self.old_x = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad):
        self.grad_bias = np.mean(grad, axis=0)
        self.grad_weights = (np.matmul(self.old_x[:, :, None], grad[:, None, :])).mean(axis=0)
        return np.dot(grad, self.weights.transpose())

    @property
    def name(self):
        n_input, n_output = np.shape(self.weights)
        return f"Linear ({n_input},{n_output})"


class ReLU():
    def forward(self, x):
        self.old_x = x
        return np.clip(x, 0, None)

    def backward(self, grad):
        return np.where(self.old_x > 0, grad, 0)

    @property
    def name(self):
        return "ReLU"


class Softmax():
    def forward(self, x):
        self.old_y = np.exp(x) / (np.exp(x).sum(axis=1)[:, None])
        return self.old_y

    def backward(self, grad):
        return self.old_y * (grad - (grad * self.old_y).sum(axis=1)[:, None])

    @property
    def name(self):
        return "Softmax"


class CrossEntropy():
    def forward(self, x, y):
        self.old_x = x.clip(min=1e-8, max=None)
        self.old_y = y
        return (np.where(y == 1, -np.log(self.old_x), 0)).sum(axis=1)

    def backward(self):
        return np.where(self.old_y == 1, -1 / self.old_x, 0)

    @property
    def name(self):
        return "Cross-Entropy"