import math
import numpy as np

SIGMA = 0
NULL = 1
RELU = 2
ELU = 3
HYPER_ELU = 0.15
MSE = 10
CROSS_ENTROPY = 11

def sigma(a):
    return 1 / (1 + math.e ** (-a))
def sigma_der(a):
    res = sigma(a) * (1 - sigma(a))
    return res

def relu(a):
    b = a.copy()
    b[a > 0] = a[a > 0]
    b[a <= 0] = 0.01 * a[a <= 0]
    return b

def relu_der(a):
    b = a.copy()
    b[a > 0] = 1
    b[a <= 0] = 0.01
    return b

def elu(a):
    b = a.copy()
    b[b < 0 ] = HYPER_ELU * (math.e ** b[b < 0] - 1)
    return b

def elu_der(a):
    b = a.copy()
    b[b > 0] = 1
    b[b < 0] = HYPER_ELU * math.e ** b[b < 0]
    return b

def null(a):
    return a

def null_der(a):
    return 1

def cross_entropy(y : np.ndarray, y_predicted : np.ndarray, **kwargs):
    res = -np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
    return res

def cross_entropy_der(y : np.ndarray, y_predicted : np.ndarray, **kwargs):
    res = -(y / y_predicted - (1 - y) / (1 - y_predicted))
    return res



def mean_square_error(y : np.ndarray, y_predicted : np.ndarray, **kwargs):
    res = ((y - y_predicted) ** 2)  / 2
    return res

def mean_square_error_der(y : np.ndarray , y_predicted : np.ndarray, **kwargs):
    res =   y_predicted - y
    return res



functions = {
    SIGMA: sigma,
    NULL: null,
    RELU: relu,
    ELU: elu
}
derivatives = {
    SIGMA: sigma_der,
    NULL: null_der,
    RELU : relu_der,
    ELU: elu_der
}

loss = {
    MSE:mean_square_error,
    CROSS_ENTROPY : cross_entropy
}
loss_der = {
    MSE: mean_square_error_der,
    CROSS_ENTROPY: cross_entropy_der
}

def function(value : np.ndarray, function_type : int):
    return functions[function_type](value)

def function_der(value : np.ndarray, function_type : int):
    return derivatives[function_type](value)