from typing import Protocol, Callable

import torch


TorchTensorTransformation = Callable[[torch.Tensor], torch.Tensor]
TorchReductionFunction = TorchTensorTransformation


class TorchLossFunction(Protocol):

    # noinspection PyShadowingBuiltins
    def __call__(self, input: torch.Tensor, target: torch.Tensor, reduction: str = 'mean'):
        pass
