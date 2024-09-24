from typing import Callable, Union, Iterable, Any

import numpy as np
import torch
from torch import optim

TensorDict = dict[str, torch.Tensor]
TensorObs = torch.Tensor | TensorDict

def detach_obs(obs: TensorObs):
    if isinstance(obs, torch.Tensor):
        return obs.detach()

    assert isinstance(obs, dict)

    return {
        key: t.detach()
        for key, t in obs.items()
    }

NpArrayDict = dict[np.ndarray]
NpObs = np.ndarray | NpArrayDict

ShapeDict = dict[str, tuple[int, ...]]

Parameters = Union[Iterable[torch.Tensor], Iterable[dict[str, Any]]]
OptimizerProvider = Callable[[Parameters], optim.Optimizer]
