import abc
from typing import Union

from torch import nn

from src.networks.core.tensor_shape import TensorShape


class Net(nn.Module, abc.ABC):

    def __init__(
            self,
            in_shape: TensorShape,
            out_shape: TensorShape,
    ):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

    def accepts_shape(self, in_shape: TensorShape) -> bool:
        for definite_symbol in self.in_shape.definite_symbols:
            if in_shape.is_definite(definite_symbol) and in_shape[definite_symbol] != self.in_shape[definite_symbol]:
                return False
        return True

    def forward_shape(self, in_shape: TensorShape) -> TensorShape:
        if not self.accepts_shape(in_shape):
            raise ValueError(f'Net ({self}) does not accept shape {in_shape}')

        return self.out_shape.evaluate_forward(in_shape)

    @staticmethod
    def as_net(module: Union['Net', nn.Module]) -> 'Net':
        if isinstance(module, Net):
            return module
        if isinstance(module, nn.Module):
            from src.networks.core.torch_net import TorchNet
            return TorchNet(module)
        raise ValueError(f'Invalid type for {module = }')
