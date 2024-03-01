import abc
from typing import Callable, SupportsFloat, Literal, Union

import torch
from torch import nn

from src.networks.net import Net

WeighingTypes = Union[float, list[SupportsFloat], torch.Tensor, "WeighingBase", Callable[[int], "WeighingBase"]]
WeighingTrainableChoices = Literal['scalar', 'vector', False, None]

class WeighingBase(Net, abc.ABC):

    @abc.abstractmethod
    def get_weight(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplemented

    def forward(self, x: torch.Tensor):
        return x * self.get_weight(x)

    @staticmethod
    def to_weighing(
            weight: WeighingTypes = 1.0,
            num_features: int = None,
            trainable: WeighingTrainableChoices = None,
    ):
        is_weight_float = isinstance(weight, float)
        is_weight_tensor_like = isinstance(weight, list) or isinstance(weight, torch.Tensor)
        is_weighing_module = isinstance(weight, WeighingBase)
        is_provider = isinstance(weight, Callable)

        assert is_weight_float or is_weight_tensor_like or is_weighing_module or is_provider
        assert trainable in ['scalar', 'vector', False, None]

        assert trainable != 'scalar' or (
                is_weight_float and
                num_features in (1, None)
        )
        assert trainable != 'vector' or (
                (is_weight_float and num_features is not None and num_features > 0) or
                (is_weight_tensor_like and num_features in (None, weight.shape[-1]))
        )
        assert not is_weighing_module or (
                trainable is None and
                num_features is None
        )
        assert not is_provider or (
                trainable is None
        )

        if trainable == 'scalar':
            return ScalarWeighing(initial_value=weight, trainable=True)

        if trainable == 'vector':
            return VectorWeighing(initial_value=weight, trainable=True)

        if is_weight_float:
            return ScalarWeighing(initial_value=weight, trainable=False)

        if is_weight_tensor_like:
            return VectorWeighing(initial_value=weight, trainable=False)

        if is_weighing_module:
            return weight

        if is_provider:
            return weight(num_features)

        raise ValueError


class ScalarWeighing(WeighingBase):

    def __init__(
            self,
            initial_value: float = 1.0,
            trainable: bool = False,
    ):
        super().__init__()

        self.w = nn.Parameter(torch.FloatTensor([initial_value]), requires_grad=trainable)

    def get_weight(self, x: torch.Tensor) -> torch.Tensor:
        return self.w


class VectorWeighing(WeighingBase):

    def __init__(
            self,
            initial_value: float | list[float] | torch.Tensor = 1.0,
            num_features: int = None,
            trainable: bool = False,
    ):
        super().__init__()

        is_val_float = isinstance(initial_value, float)
        is_val_tensor_like = isinstance(initial_value, list) or isinstance(initial_value, torch.Tensor)

        assert not isinstance(initial_value, float) or num_features is not None
        assert not is_val_float or (num_features is not None and num_features > 0)
        assert not is_val_tensor_like or num_features in (None, initial_value.shape[-1])

        if isinstance(initial_value, float):
            initial_value = torch.ones(num_features) * initial_value
        if isinstance(initial_value, list):
            initial_value = torch.FloatTensor(initial_value)

        self.w = nn.Parameter(initial_value, requires_grad=trainable)

    def get_weight(self, x: torch.Tensor) -> torch.Tensor:
        return self.w


class ModuleWeighing(WeighingBase):

    def __init__(
            self,
            module: nn.Module,
    ):
        super().__init__()
        self.module = module

    def get_weight(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)
