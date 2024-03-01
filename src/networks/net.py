import abc
from typing import Literal, Union, Iterable

from torch import nn


class Net(nn.Module, abc.ABC):

    IN_FEATURES_ANY = 'any'
    OUT_FEATURES_SAME = 'same'

    def __init__(self, in_features: int | Literal['any'], out_features: int | Literal['same']):
        super().__init__()

        assert isinstance(in_features, int) or in_features == self.IN_FEATURES_ANY
        assert isinstance(out_features, int) or out_features == self.OUT_FEATURES_SAME

        self.in_features = in_features.lower() if isinstance(in_features, str) else in_features
        self.out_features = out_features.lower() if isinstance(out_features, str) else out_features

    def actualize_out_features(self, in_features: int) -> int:
        if self.in_features != Net.IN_FEATURES_ANY and self.in_features != in_features:
            raise ValueError('Argument in_features must be same as attribute self.in_features '
                             'if self.in_features is not equal to "any"')
        if self.out_features == Net.OUT_FEATURES_SAME:
            return in_features
        return self.out_features

    def are_in_out_features_defined(self):
        return self.in_features != self.IN_FEATURES_ANY and self.out_features != self.OUT_FEATURES_SAME

    @classmethod
    def are_all_in_out_features_defined(cls, nets: Iterable['Net']):
        return all(
            net.in_features != cls.IN_FEATURES_ANY and net.out_features != cls.OUT_FEATURES_SAME
            for net in nets
        )

    @staticmethod
    def as_net(module: Union['Net', nn.Module]) -> 'Net':
        if isinstance(module, Net):
            return module
        return TorchModuleNet(module)


from src.networks.torch_module_net import TorchModuleNet
