import numpy as np
import torch

TensorOrNpArray = torch.Tensor | np.ndarray


def format_summary_statics(
        arr: TensorOrNpArray,
        mean_format: str | None = '.2f',
        std_format: str | None = '.2f',
        min_value_format: str | None = '.2f',
        max_value_format: str | None = '.2f',
):
    mean, std, min_value, max_value = compute_summary_statistics(arr)

    representation = ''

    if mean_format:
        representation += mean.__format__(mean_format)

    if std_format:
        representation += f' ± {std.__format__(std_format)}'

    if min_value_format or max_value_format:
        representation += ' '

        if min_value_format:
            representation += f'[{min_value.__format__(min_value_format)},'
        else:
            representation += f'(,'

        if max_value_format:
            representation += f' {max_value.__format__(max_value_format)}]'
        else:
            representation += f')'

    return representation



def compute_summary_statistics(
        arr: TensorOrNpArray
) -> tuple[TensorOrNpArray, TensorOrNpArray, TensorOrNpArray, TensorOrNpArray]:
    mean = arr.mean()
    std = arr.std()
    min_value = arr.min()
    max_value = arr.max()

    return mean, std, min_value, max_value
