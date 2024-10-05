from typing import NamedTuple

import torch

from src.reinforcement_learning.core.buffers.base_buffer import BaseBuffer
from src.reinforcement_learning.core.type_aliases import TensorObs


class ReservoirBufferSamples(NamedTuple):
    observations: TensorObs
    actions: torch.Tensor
    next_observations: TensorObs
    dones: torch.Tensor
    rewards: torch.Tensor

class BaseReservoirBuffer(BaseBuffer):
    def add(self, *args, **kwargs) -> None:
