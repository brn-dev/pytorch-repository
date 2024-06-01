import numpy as np


def softmax(arr: np.ndarray, temperature: float = 1.0, normalize: bool = False):
    if normalize:
        arr -= arr.min()
        arr /= arr.max()

    exp_arr = np.exp(arr / temperature)
    return exp_arr / exp_arr.sum()
