import numpy as np
from functions import CrossEntropy

class Model():
    def __init__(self, layers, cost = CrossEntropy()):
        self.layers = layers
        self.cost = cost

    def forward(self, x):
        #print(f"Before processing: {np.shape(x)}")

        for layer in self.layers:
            x = layer.forward(x)
        #    print(f"After passing through {layer} : {np.shape(x)}")
        return x
    
    def predict(self, x):
        return self.forward(x)

    def loss(self, x, y):
        l = self.cost.forward(self.forward(x), y)
        #print(f"After passing through {self.cost} : {np.shape(l)}")
        return l

    def backward(self):
        grad = self.cost.backward()
        #print(f"After backward on {self.cost} : {np.shape(grad)}")

        for i in range(len(self.layers) - 1, -1, -1):
            grad = self.layers[i].backward(grad)
         #   print(f"After backward on {self.layers[i]} : {np.shape(grad)}")