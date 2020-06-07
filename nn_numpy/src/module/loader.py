from json_params import *
from models import *
from functions import *

filename = "../data/nn_params.json"
paramsHandler = JsonParams(filename)

layers = [Linear(784, 50), ReLU(), Linear(50, 10), Softmax()]
layers = paramsHandler.loadParamsIntoLayers(layers)

model = Model(layers = layers)
