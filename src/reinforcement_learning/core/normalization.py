from enum import Enum

import numpy as np
import torch


class NormalizationType(Enum):
    Mean = 0
    Std = 1
    MeanAndStd = 2


def normalize_tensor(tensor: torch.Tensor, normalization_type: NormalizationType) -> torch.Tensor:
    match normalization_type:
        case NormalizationType.Mean:
            return tensor - tensor.mean()
        case NormalizationType.Std:
            return tensor / (tensor.std() + 1e-6)
        case NormalizationType.MeanAndStd:
            return (tensor - tensor.mean()) / (tensor.std() + 1e-6)
        case _:
            raise ValueError(f'Unknown normalization_type {normalization_type}')


def normalize_np_array(array: np.ndarray, normalization_type: NormalizationType) -> np.ndarray:
    match normalization_type:
        case NormalizationType.Mean:
            return array - array.mean()
        case NormalizationType.Std:
            return array / (array.std() + 1e-6)
        case NormalizationType.MeanAndStd:
            return (array - array.mean()) / (array.std() + 1e-6)
        case _:
            raise ValueError(f'Unknown normalization_type {normalization_type}')
