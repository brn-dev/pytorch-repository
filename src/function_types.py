from typing import Protocol, Callable

import torch


TorchTensorFn = Callable[[torch.Tensor], torch.Tensor]


class TorchLossFn(Protocol):

    # noinspection PyShadowingBuiltins
    def __call__(self, input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean'):
        pass
