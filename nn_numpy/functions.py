import numpy as np

class Linear():
    def __init__(self, n_input, n_output):
        self.weights = np.random.randn(n_input, n_output)
        self.bias = np.zeros(n_output)

    def forward(self, x):
        self.old_x = x
        return np.matmul(x, self.weights) + self.bias

    # There will be a gradient wrt each output of this layer that comes from layers ahead
    # Therefore shape of grad will be the (n_samples, n_outputs) because that many
    # elements can come inside from ahead in the layers.
    def backward(self, grad):
        # This is averaging over all rows. Thus shape is (n_outputs,)
        self.grad_bias = np.mean(grad, axis=0)

        # x : (n_samples, n_inputs), grad: (n_samples, n_outputs)
        # "None" adds a unit axis whereexver specified.
        # Therefore matrix multiplication becomes (n_samples, n_inputs, 1) and (n_samples, 1, n_outputs)
        # This can be translated to: for all "samples" do (n_inputs, 1) * (1, n_outputs)
        # This gives for all "samples", (n_inputs, n_outputs) ie (n_samples, n_inputs, n_outputs)

        # (n_inputs, n_outputs) should be expected because, in linear layer, each input node touches
        # all the output nodes.
        # Taking average at axis 0 will be taking average across all samples since "n_samples" is
        # the 0th axis
        # As a result shape of weights is (n_inputs, n_outputs)
        self.grad_weights = (np.matmul(self.old_x[:, :, None], grad[:, None, :])).mean(axis=0)

        # (n_samples, n_outputs) * (n_inputs, n_outputs)T will be (n_samples, n_inputs)
        # This is expected because the previous layer this layer will pass it to, will have
        # n_inputs number of "outputs" in its layer, which will form (n_samples, n_outputs) for that layer
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