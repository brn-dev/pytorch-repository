import numpy as np


def softmax(arr: np.ndarray, temperature: float = 1.0, normalize: bool = False, eps: float = 1e-6):
    if normalize:
        arr -= arr.min()
        max_ = arr.max()
        if max_ > eps:  # don't divide by 0
            arr /= max_

    exp_arr = np.exp(arr / temperature)
    return exp_arr / exp_arr.sum()
