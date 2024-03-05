import abc
from typing import Union

from torch import nn

from src.networks.tensor_shape import TensorShape


class Net(nn.Module, abc.ABC):

    def __init__(
            self,
            in_shape: TensorShape,
            out_shape: TensorShape,
    ):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

    @staticmethod
    def as_net(module: Union['Net', nn.Module]) -> 'Net':
        if isinstance(module, Net):
            return module
        if isinstance(module, nn.Module):
            return TorchNet(module)
        raise ValueError(f'Invalid type for {module = }')


from src.networks.torch_net import TorchNet
