import numpy as np


def softmax(arr: np.ndarray, temperature: float = 1.0):
    exp_arr = np.exp(arr / temperature)
    return exp_arr / exp_arr.sum()
