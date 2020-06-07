from json_params import *

class Trainer():
    def __init__(self, model = None, inputs = None, labels = None, epochs = 1, learning_rate = 0.1):
        self.model = model
        self.inputs = inputs
        self.labels = labels
        self.epochs = epochs
        self.learning_rate = learning_rate

    def setModel(self, model):
        self.model = model

    def setInputs(self, inputs):
        self.inputs = inputs
    
    def setLabels(self, labels):
        self.labels = labels

    def setEpochs(self, epochs):
        self.epochs = epochs

    def setLearningRate(self, learning_rate):
        self.learning_rate = learning_rate

    def run(self, logging = False):
        for i in range(self.epochs):
            # forward pass
            l = self.model.loss(self.inputs, self.labels).sum()

            # backward pass
            self.model.backward()

            # update params
            for layer in self.model.layers:
                if type(layer) is Linear:
                    layer.weights -= self.learning_rate * layer.grad_weights
                    layer.bias -= self.learning_rate * layer.grad_bias
                    
            if logging is True:
                print(f"total loss: {l}, inputs: {np.shape(inputs)[0]}, average loss: {l/np.shape(inputs)[0]}")

    def saveParams(self, filename):
        paramsHandler = JsonParams(filename)
        paramsHandler.dumpParamsInJson(self.model)