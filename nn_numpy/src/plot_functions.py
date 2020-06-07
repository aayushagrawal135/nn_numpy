import matplotlib.pyplot as plt

def show(img, title = None):
    plt.imshow(img, cmap = "gray")
    if title is not None:
        plt.title(title)