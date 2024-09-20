from typing import Callable, Union, Iterable, Any

import numpy as np
import torch
from torch import optim

TensorDict = dict[str, torch.Tensor]
TensorObs = torch.Tensor | TensorDict

NpArrayDict = dict[np.ndarray]
NpObs = np.ndarray | NpArrayDict

ShapeDict = dict[str, tuple[int, ...]]

Parameters = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]
OptimizerProvider = Callable[[Parameters], optim.Optimizer]
