import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from functions import *
from models import *
from metrics import *
from json_params import *

# The data has been taken from below
# Set appropriate path in your local or run on kaggle itself with the jupyter notebook present 
# https://www.kaggle.com/oddrationale/mnist-in-csv

train_path = "mnist_train.csv"
data = pd.read_csv(train_path)

def featLabelSep(data):
    labels = data.iloc[:, 0:1].to_numpy()
    pixels = data.iloc[:, 1:].to_numpy()
    return pixels, labels

pixels, labels = featLabelSep(data)

def oneHotEncode(labels):
    rows = np.shape(labels)[0]
    cols = np.shape(np.unique(labels))[0]
    base = np.zeros((rows, cols), dtype = int)
    
    for index, value in enumerate(labels):
        base[index][value] = 1
        
    return base

def split_vals(a, n):
    return a[:n], a[n:]

trn = 100
trn_labels, valid_labels = split_vals(labels, trn)
trn_pixels, valid_pixels = split_vals(pixels, trn)

def normalise(a, mean, std):
    return (a - mean)/std

mean = trn_pixels.mean()
std = trn_pixels.std()
norm_trn_pixels = normalise(trn_pixels, mean, std)

"""
def show(img, title = None):
    plt.imshow(img, cmap = "gray")
    if title is not None:
        plt.title(title)
        
sample_to_plot = np.reshape(trn_pixels[0], (28, 28))
show(sample_to_plot, trn_labels[0])
"""

layers = [Linear(784, 50), ReLU(), Linear(50, 10), Softmax()]
cost = CrossEntropy()
model = Model(layers = layers, cost = cost)

epochs = 20
learning_rate = 1

def train(model, inputs, labels, epochs=1, learning_rate=0.1):
    for i in range(epochs):
        l = model.loss(inputs, labels).sum()

        model.backward()

        for layer in model.layers:
            if type(layer) is Linear:
                layer.weights -= learning_rate * layer.grad_weights
                layer.bias -= learning_rate * layer.grad_bias
                
        print(f"total loss: {l}, inputs: {np.shape(inputs)[0]}, average loss: {l/np.shape(inputs)[0]}")

train(model = model, inputs = norm_trn_pixels, labels = oneHotEncode(trn_labels), epochs = epochs)


filename = "nn_params.json"
paramsHandler = JsonParams(filename)

paramsHandler.dumpParamsInJson(model)

layersR = [Linear(784, 50), ReLU(), Linear(50, 10), Softmax()]
layersR = paramsHandler.loadParamsIntoLayers(layersR)

modelR = Model(layers = layersR, cost = cost)
p = modelR.predict(norm_trn_pixels[:2])
print(np.argmax(p, axis = 1))
