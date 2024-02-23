from dataclasses import dataclass, asdict
from typing import TypeVar

from torch import nn

@dataclass(init=True)
class HyperParameters:

    def __set_item__(self, key, item):
        raise TypeError('HyperParameters are frozen, cannot set values')

    def __getitem__(self, key: str):
        return getattr(self, key)

    def to_dict(self):
        return asdict(self)


HP = TypeVar('HP', bound=HyperParameters)

class NNBase(nn.Module):

    @classmethod
    def from_hyper_parameters(cls, hyper_parameters: HP):
        return cls(**hyper_parameters.to_dict())
