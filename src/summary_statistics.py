import numpy as np
import torch

TensorOrNpArray = torch.Tensor | np.ndarray


def format_summary_statics(
        arr: TensorOrNpArray,
        mean_format: str | None = '.2f',
        std_format: str | None = '.2f',
        min_value_format: str | None = None,
        max_value_format: str | None = None,
        n_format: str | None = None
):
    mean, std, min_value, max_value = compute_summary_statistics(arr)

    representation = ''

    if mean_format:
        representation += mean.__format__(mean_format)

    if std_format:
        representation += f' ± {std.__format__(std_format)}'

    if min_value_format and max_value_format:
        representation += f' [{min_value.__format__(min_value_format)}, {max_value.__format__(max_value_format)}]'
    elif min_value_format:
        representation += f' ≥ {min_value.__format__(min_value_format)}'
    elif max_value_format:
        representation += f' ≤ {max_value.__format__(max_value_format)}'

    if n_format:
        representation += f' (n={len(arr).__format__(n_format)})'

    return representation


def compute_summary_statistics(
        arr: TensorOrNpArray
) -> tuple[TensorOrNpArray, TensorOrNpArray, TensorOrNpArray, TensorOrNpArray]:
    mean = arr.mean()
    std = arr.std()
    min_value = arr.min()
    max_value = arr.max()

    return mean, std, min_value, max_value
