import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from functions import *
from models import *
from metrics import *

# The data has been taken from below
# Set appropriate path in your local or run on kaggle itself with the jupyter notebook present 
# https://www.kaggle.com/oddrationale/mnist-in-csv

train_path = "mnist_train.csv"

data = pd.read_csv(train_path)
labels = data.iloc[:, 0:1].to_numpy()
pixels = data.iloc[:, 1:].to_numpy()

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

def dumpParamsInJson(model):
    params = dict()
    filename = "nn_params.json"

    for layer in model.layers:
        if type(layer) is Linear:
            params[f"{layer}_weights"] = layer.weights.tolist()
            params[f"{layer}_bias"] = layer.bias.tolist()
            
    with open(filename, "w") as file:
        json.dump(params, file)
    
    return params 

dumpParamsInJson(model)

def loadParamsIntoLayers(filename, layers):

    with open(filename, "r") as file:
        params = json.load(file)
    
    for layer in layers:
        if type(layer) is Linear:
            key_weight = f"{layer}_weights"
            key_bias = f"{layer}_bias"
            layer.setWeights(np.array(params[key_weight]))
            layer.setBias(np.array(params[key_bias]))
    
    return layers

filename = "nn_params.json"
layersR = [Linear(784, 20), ReLU(), Linear(20, 10), Softmax()]
layersR = loadParamsIntoLayers(filename, layers)

modelR = Model(layers = layersR, cost = cost)

p = modelR.predict(norm_trn_pixels[:2])
