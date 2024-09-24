from typing import TypedDict, Optional, NotRequired

import numpy as np
import torch

TensorOrNpArray = torch.Tensor | np.ndarray

SMALL_DATA_THRESHOLD = 8


class SummaryStatistics(TypedDict):
    n: int
    mean: float
    std: NotRequired[float]
    min_value: NotRequired[float]
    max_value: NotRequired[float]
    data: NotRequired[list]


def format_summary_statics(
        x: TensorOrNpArray | SummaryStatistics | None,
        mean_format: str | None = '.2f',
        std_format: str | None = '.2f',
        min_value_format: str | None = None,
        max_value_format: str | None = None,
        n_format: str | None = None
):
    if isinstance(x, dict):
        summary_statistics = x
    else:
        summary_statistics = compute_summary_statistics(x)

    if summary_statistics is None:
        return 'N/A'

    n = summary_statistics['n']
    mean = summary_statistics['mean']
    std = summary_statistics['std']
    min_value = summary_statistics['min_value']
    max_value = summary_statistics['max_value']

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
        representation += f' (n={n.__format__(n_format)})'

    return representation


def compute_summary_statistics(
        arr: TensorOrNpArray,
        small_data_threshold: int = SMALL_DATA_THRESHOLD
) -> Optional[SummaryStatistics]:
    if isinstance(arr, np.ndarray):
        n = arr.size
    else:
        n = arr.numel()

    if n == 0:
        return None

    mean = arr.ravel().mean().item()

    if n == 1:
        return {
            'n': n,
            'mean': mean,
        }

    summary_stats: SummaryStatistics = {
        'n': n,
        'mean': mean,
        'std': arr.ravel().std().item(),
        'min_value': arr.min().item(),
        'max_value': arr.max().item(),
    }

    if n <= small_data_threshold:
        summary_stats['data'] = arr.tolist()

    return summary_stats
