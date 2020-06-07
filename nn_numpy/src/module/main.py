import numpy as np
import pandas as pd

from functions import *
from models import *
from metrics import *
from json_params import *
from preprocessing import *
from trainer import *
from plot_functions import *

# The data has been taken from below
# Set appropriate path in your local or run on kaggle itself with the jupyter notebook present 
# https://www.kaggle.com/oddrationale/mnist-in-csv
train_path = "../data/mnist_train.csv"
data = pd.read_csv(train_path)

pixels, labels = featLabelSep(data)

trn = 100
trn_labels, valid_labels = split_vals(labels, trn)
trn_pixels, valid_pixels = split_vals(pixels, trn)

mean = trn_pixels.mean()
std = trn_pixels.std()
norm_trn_pixels = normalise(trn_pixels, mean, std)
norm_valid_pixels = normalise(valid_pixels, mean, std)
    
# sample_to_plot = np.reshape(trn_pixels[0], (28, 28))
# show(sample_to_plot, trn_labels[0])

layers = [Linear(784, 50), ReLU(), Linear(50, 10), Softmax()]
cost = CrossEntropy()
model = Model(layers = layers, cost = cost)

trainer = Trainer(model=model)
trainer.setInputs(norm_trn_pixels)
trainer.setLabels(oneHotEncode(trn_labels))
trainer.setEpochs(20)
trainer.setLearningRate(1)
trainer.run()

filename = "../data/nn_params.json"
paramsHandler = JsonParams(filename)
trainer.saveParams(filename)
