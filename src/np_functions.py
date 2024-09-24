import numpy as np


def softmax(arr: np.ndarray, temperature: float = 1.0, normalize: bool = False, eps: float = 1e-6):
    if normalize:
        arr = arr.copy()
        arr -= arr.min()
        max_ = arr.max()
        if max_ > eps:  # don't divide by 0
            arr /= max_

    exp_arr = np.exp(arr / temperature)
    return exp_arr / exp_arr.sum()

def symmetric_log(arr: np.ndarray):
    arr = arr.copy()

    positive = arr > 0
    arr[positive] = np.log(arr[positive] + 1)

    negative = arr < 0
    arr[negative] = -np.log(-arr[negative] + 1)

    return arr

def inv_symmetric_log(arr: np.ndarray):
    arr = arr.copy()

    positive = arr > 0
    arr[positive] = np.exp(arr[positive]) -1

    negative = arr < 0
    arr[negative] = -np.exp(-arr[negative]) + 1

    return arr

