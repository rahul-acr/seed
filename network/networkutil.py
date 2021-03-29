import numpy as np
from scipy.special import expit

sigmoid_activation = 'sigmoid'
tanh_activation = 'tanh'
softmax_activation = 'softmax'

# compute sigmoid
def sigmoid(x):
    return expit(x)


# convert output of sigmoid function to its derivative
def sigmoid_output_to_derivative(output):
    return output * (1 - output)


# compute tanh
def tanh(x):
    return np.tanh(x)


# convert output of tanh function to its derivative
def tanh_output_to_derivative(output):
    return 1 - np.square(output)


# compute softmax
def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)


# convert output of tanh function to its derivative
def softmax_output_to_derivative(output):
    return output * (1 - output)