import abc
from typing import Union, Literal, Callable

from torch import nn

from src.module_analysis import count_parameters, get_gradients_per_parameter
from src.networks.core.tensor_shape import TensorShape, TensorShapeError


# TODO: extend jit.ScriptModule instead?
class Net(nn.Module, abc.ABC):

    def __init__(
            self,
            in_shape: TensorShape,
            out_shape: TensorShape,
            allow_extra_dimensions: bool = True,
    ):
        super().__init__()

        self.in_shape = in_shape
        self.out_shape = out_shape

        self.allow_extra_dimensions = allow_extra_dimensions

    def check_in_shape(self, in_shape: TensorShape):
        if not self.allow_extra_dimensions and self.in_shape.dimension_names < in_shape.dimension_names:
            raise TensorShapeError(f'This net only accepts tensors with the following dimensions: '
                                   f'{self.in_shape.dimension_names}, got {in_shape.dimension_names}',
                                   self_in_shape=self.in_shape, in_shape=in_shape)

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

    def count_parameters(self, requires_grad_only: bool = True):
        return count_parameters(self, requires_grad_only)

    def get_gradients_per_layer(self, param_type: Literal['all', 'weight', 'bias'] = 'all'):
        return get_gradients_per_parameter(self, param_type)

    @staticmethod
    def as_net(module: Union['Net', nn.Module]) -> 'Net':  # This has to be Union, string type and | does not work
        if isinstance(module, Net):
            return module
        if isinstance(module, nn.Module):
            from src.networks.core.torch_wrappers.torch_net import TorchNet
            return TorchNet.wrap(module)
        raise ValueError(f'Invalid type for {module = }')

    @staticmethod
    def seq_as_net(*modules: Union['Net', nn.Module]):  # This has to be Union, string type and | does not work
        return Net.as_net(nn.Sequential(*modules))

    @staticmethod
    def provider_seq_as_net(
            in_features: int,
            *providers: Callable[[int], Union['Net', nn.Module]]
    ) -> 'Net':
        current_shape = TensorShape(features=in_features)

        nets = []

        for i, provider in enumerate(providers):
            try:
                net = provider(current_shape.get_definite_features())
                nets.append(net)
                current_shape = Net.as_net(net).forward_shape(current_shape)
            except TensorShapeError as tse:
                raise TensorShapeError(
                    f'TensorShapeError occurred while forwarding shape through layer {i}',
                    parent_error=tse
                )

        return Net.seq_as_net(*nets)

