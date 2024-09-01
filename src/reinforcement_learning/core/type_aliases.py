from typing import Callable

import numpy as np
import torch
from torch import nn

TensorDict = dict[str, torch.Tensor]
TensorObs = torch.Tensor | TensorDict
TensorObsPreprocessing = Callable[[TensorObs], torch.Tensor] | nn.Module

NpArrayDict = dict[np.ndarray]
NpObs = np.ndarray | NpArrayDict

ShapeDict = dict[str, tuple[int, ...]]
