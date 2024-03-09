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
        for definite_dim in self.in_shape.definite_dimension_names:
            if in_shape.is_definite(definite_dim) and in_shape[definite_dim] != self.in_shape[definite_dim]:
                return False
        return True

    def forward_shape(self, in_shape: TensorShape) -> TensorShape:
        if not self.accepts_shape(in_shape):
            raise ValueError(f'Net ({self}) does not accept shape {in_shape}')

        return self.out_shape.evaluate_forward(in_shape)

    def modifies_shape_in_dimension(self, dim_key: str, in_shape: TensorShape = None):
        if in_shape is None:
            in_shape = self.in_shape

        out_shape = self.forward_shape(in_shape)

        return in_shape[dim_key] != out_shape[dim_key]

    @staticmethod
    def as_net(module: Union['Net', nn.Module]) -> 'Net':
        if isinstance(module, Net):
            return module
        if isinstance(module, nn.Module):
            from src.networks.core.torch_net import TorchNet
            return TorchNet.wrap(module)
        raise ValueError(f'Invalid type for {module = }')
