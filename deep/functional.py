import numpy as np

# === ACTIVATIONS === #
def relu(inputs):
    return np.maximum(0, inputs)

def tanh(inputs):
    return np.tanh(inputs)

def softmax(inputs):
    exp_val = np.exp(inputs - np.max(inputs))
    return exp_val/exp_val.sum(axis=0)

def sigmoid(inputs):
    return 1/(1+np.exp(-inputs))

def one_hot_encode(labels: np.ndarray):
    length = max(labels)+1
    one_hot_vector = []

    for idx, label in enumerate(labels):
        vector = np.zeros(length)
        vector[label] = 1
        one_hot_vector.append(vector)
        
    return np.array(one_hot_vector)
