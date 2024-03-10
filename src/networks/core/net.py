import abc
from typing import Union

from torch import nn

from src.networks.core.tensor_shape import TensorShape, TensorShapeError


class Net(nn.Module, abc.ABC):

    def __init__(
            self,
            in_shape: TensorShape,
            out_shape: TensorShape,
    ):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

    def check_in_shape(self, in_shape: TensorShape):
        for definite_dim in self.in_shape.definite_dimension_names:
            if in_shape.is_definite(definite_dim) and in_shape[definite_dim] != self.in_shape[definite_dim]:
                raise TensorShapeError(f'This net only accepts tensors with size {self.in_shape[definite_dim]} '
                                       f'in dimension {definite_dim}. in_shape has size {in_shape[definite_dim]}',
                                       self_in_shape=self.in_shape, in_shape=in_shape)

    def accepts_in_shape(self, in_shape: TensorShape) -> tuple[bool, TensorShapeError | None]:
        accepts_in_shape, error = True, None
        try:
            self.check_in_shape(in_shape)
        except TensorShapeError as tse:
            accepts_in_shape, error = False, tse
        return accepts_in_shape, error

    def forward_shape(self, in_shape: TensorShape) -> TensorShape:
        self.check_in_shape(in_shape)

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
