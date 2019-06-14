import numpy as np

def sigmoid(val):
    val = np.array(val)
    return (1/(1+np.exp(-1 * val)))

