import abc
from typing import Callable, SupportsFloat, Literal, Union

import torch
from torch import nn

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape

WeighingTypes = Union[float, list[SupportsFloat], torch.Tensor, "Weighing", Callable[[int], "Weighing"]]
WeighingTrainableChoices = Literal['scalar', 'vector', False, None]


class Weighing(Net, abc.ABC):

    @abc.abstractmethod
    def get_weight(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def forward(self, x: torch.Tensor):
        return x * self.get_weight(x)

    @staticmethod
    def to_weighing(
            num_features: int,
            weight: WeighingTypes = 1.0,
            trainable: WeighingTrainableChoices = None,
    ):
        is_weight_float = isinstance(weight, float)
        is_weight_tensor_like = isinstance(weight, list) or isinstance(weight, torch.Tensor)
        is_weighing = isinstance(weight, Weighing)
        is_weighing_provider = isinstance(weight, Callable)

        assert is_weight_float or is_weight_tensor_like or is_weighing or is_weighing_provider
        assert trainable in ['scalar', 'vector', False, None]

        assert trainable != 'scalar' or (
                is_weight_float and
                num_features in (1, None)
        )
        assert trainable != 'vector' or (
                (is_weight_float and num_features is not None and num_features > 0) or
                (is_weight_tensor_like and num_features in (None, weight.shape[-1]))
        )
        assert not is_weighing or (
                trainable is None and
                num_features is None
        )
        assert not is_weighing_provider or (
                trainable is None
        )

        if is_weighing:
            return weight

        if is_weighing_provider:
            return weight(num_features)

        if trainable == 'scalar':
            return ScalarWeighing(initial_value=weight, trainable=True)

        if trainable == 'vector':
            return VectorWeighing(initial_value=weight, trainable=True)

        if is_weight_float:
            return ScalarWeighing(initial_value=weight, trainable=False)

        if is_weight_tensor_like:
            return VectorWeighing(initial_value=weight, trainable=False)

        raise ValueError


class ScalarWeighing(Weighing):

    def __init__(
            self,
            initial_value: float = 1.0,
            trainable: bool = False,
    ):
        super().__init__(
            in_shape=TensorShape(),
            out_shape=TensorShape(),
        )

        self.w = nn.Parameter(torch.tensor([initial_value]), requires_grad=trainable)

    def get_weight(self, x: torch.Tensor) -> torch.Tensor:
        return self.w


class VectorWeighing(Weighing):

    def __init__(
            self,
            num_features: int = None,
            initial_value: float | list[float] | torch.Tensor = 1.0,
            trainable: bool = False,
    ):
        is_val_float = isinstance(initial_value, float)
        is_val_tensor_like = isinstance(initial_value, list) or isinstance(initial_value, torch.Tensor)

        assert not is_val_float or (num_features is not None and num_features > 0)
        assert not is_val_tensor_like or num_features in (None, initial_value.shape[-1])

        if is_val_float:
            initial_value = torch.ones(num_features) * initial_value
        if is_val_tensor_like:
            initial_value = torch.tensor(initial_value)

        self.num_features = len(initial_value)

        tensor_shape = TensorShape(features=self.num_features)
        super().__init__(
            in_shape=tensor_shape,
            out_shape=tensor_shape.copy(),
        )

        self.w = nn.Parameter(initial_value, requires_grad=trainable)

    def get_weight(self, x: torch.Tensor) -> torch.Tensor:
        return self.w


class NetWeighing(Weighing):

    def __init__(
            self,
            net: Net,
    ):
        super().__init__(
            in_shape=net.in_shape,
            out_shape=net.out_shape,
        )
        self.net = net

    def get_weight(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
