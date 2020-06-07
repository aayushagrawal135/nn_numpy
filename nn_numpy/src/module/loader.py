from json_params import *
from models import *
from functions import *

filename = "../data/nn_params.json"
paramsHandler = JsonParams(filename)

layers = [Linear(784, 50), ReLU(), Linear(50, 10), Softmax()]
layers = paramsHandler.loadParamsIntoLayers(layers)

model = Model(layers = layers)
#p = model.predict(norm_trn_pixels[:2])
#print(np.argmax(p, axis = 1))