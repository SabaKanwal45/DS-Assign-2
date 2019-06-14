import numpy as np


def add_bias_term_in_data(train_data):
    x = np.array(train_data)
    y = []
    for i in range(len(train_data)):
        y.append([1])
    train_data_with_bias = np.append(y, x, axis=1)
    return train_data_with_bias