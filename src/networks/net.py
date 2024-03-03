import abc
from typing import Literal, Union, Iterable

from torch import nn


class Net(nn.Module, abc.ABC):

    IN_FEATURES_ANY = 'any'
    OUT_FEATURES_SAME = 'same'

    InFeaturesType = int | Literal['any']
    OutFeaturesType = int | Literal['same']

    in_features: InFeaturesType
    out_features: OutFeaturesType

    @property
    def in_features_defined(self) -> bool:
        return isinstance(self.in_features, int)

    @property
    def out_features_defined(self) -> bool:
        return isinstance(self.out_features, int)

    @property
    def in_out_features_defined(self) -> bool:
        return self.in_features_defined and self.out_features_defined

    @property
    def in_out_features_same(self) -> bool:
        return self.out_features == Net.OUT_FEATURES_SAME \
               or self.in_features_defined and self.in_features == self.out_features

    def __init__(
            self,
            in_features: InFeaturesType,
            out_features: OutFeaturesType,
            allow_undefined_in_out_features: bool = False,
    ):
        super().__init__()

        assert isinstance(in_features, int) or in_features == self.IN_FEATURES_ANY
        assert isinstance(out_features, int) or out_features == self.OUT_FEATURES_SAME

        self.in_features = in_features
        self.out_features = out_features

        if self.in_features_defined and self.in_out_features_same:
            self.out_features = in_features

        assert allow_undefined_in_out_features or self.in_out_features_defined

    def accepts_any_in_features(self) -> bool:
        return self.in_features == self.IN_FEATURES_ANY

    def accepts_in_features(self, in_features: int) -> bool:
        return self.accepts_any_in_features() or self.in_features == in_features

    def actualize_out_features(self, in_features: int) -> int:
        if not self.accepts_in_features(in_features):
            raise ValueError('Argument in_features must be same as attribute self.in_features '
                             'if self.in_features is not equal to "any"')

        if self.out_features == self.OUT_FEATURES_SAME:
            return in_features
        return self.out_features

    @staticmethod
    def as_net(module: Union['Net', nn.Module]) -> 'Net':
        if isinstance(module, Net):
            return module
        if isinstance(module, nn.Module):
            return TorchNet(module)
        raise ValueError(f'Invalid type for {module = }')


from src.networks.torch_net import TorchNet
