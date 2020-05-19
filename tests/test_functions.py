import unittest
import numpy as np
import sys

sys.path.append("../")
from nn_numpy.functions import *

def get_input(a1, a2):
    return np.random.randn(a1, a2)

def get_grad(a1, a2):
    return np.random.randn(a1, a2)

class TestLinearLayer(unittest.TestCase):
    
    # test params: weights, bias
    def test_param_shapes(self):
        n_input = 784
        n_output = 50
        linear = Linear(n_input, n_output)
        
        self.assertEqual(np.shape(linear.weights), (n_input, n_output))
        self.assertEqual(np.shape(linear.bias), (n_output,))

    # test forward pass
    def test_forward_shapes(self):
        n_input = 784
        n_output = 50
        n_samples = 100
        x = get_input(n_samples, n_input)
        linear = Linear(n_input, n_output)
        y = linear.forward(x)

        self.assertEqual(np.shape(y), (n_samples, n_output))

    # test backward pass
    def test_backward_shapes(self):
        n_input = 784
        n_output = 50
        n_samples = 100
        x = get_input(n_samples, n_input)
        linear = Linear(n_input, n_output)

        grad = get_grad(n_samples, n_output)
        linear.forward(x)
        grad_passed_behind = linear.backward(grad)
        
        self.assertEqual(np.shape(linear.grad_bias), (n_output,))
        self.assertEqual(np.shape(linear.grad_weights), (n_input, n_output))
        self.assertEqual(np.shape(grad_passed_behind), (n_samples, n_input))
