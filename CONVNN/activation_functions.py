import numpy as np

################ ACTIVATION FUNCTIONS ##########################

def sigmoid(inp):
    if type(inp) in [list, tuple]:
        inp = np.array(inp)
    sig = 1.0 / (1 + np.exp(-1 * inp))
    return sig

def relu(inp):
    if not(type(inp) in [tuple, list, np.ndarray]):
        if inp < 0:
            return 0
        else:
            return inp
    elif type(inp) in [list, tuple]:
        inp = np.array(inp)
    relu = inp
    relu[inp<0] = 0
    return relu

def tanh(inp):
    if type(inp) in [list, tuple]:
        inp = np.array(inp)
    exn = np.exp(-inp)
    exp = np.exp(inp)
    tanh = (exp - exn)/ (exp + exn)
    return tanh   

def softmax(inp):
    ex = np.exp(inp)
    sm = ex / (np.sum(ex))
    return sm

################################################################
