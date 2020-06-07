def accuracy(predictions, labels):
    p = predictions.argmax(axis = 1)
    l = labels.reshape(-1,)
    return (p == l).sum()/np.shape(p)[0]