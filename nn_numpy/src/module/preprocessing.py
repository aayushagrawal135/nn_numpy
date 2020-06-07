import numpy as np

def featLabelSep(data):
    labels = data.iloc[:, 0:1].to_numpy()
    pixels = data.iloc[:, 1:].to_numpy()
    return pixels, labels

def oneHotEncode(labels):
    rows = np.shape(labels)[0]
    cols = np.shape(np.unique(labels))[0]
    base = np.zeros((rows, cols), dtype = int)
    
    for index, value in enumerate(labels):
        base[index][value] = 1
        
    return base

def split_vals(a, n):
    return a[:n], a[n:]

def normalise(a, mean, std):
    return (a - mean)/std