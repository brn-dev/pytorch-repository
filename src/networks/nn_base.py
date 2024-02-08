from typing import TypeVar

from torch import nn

from src.hyper_parameters import HyperParameters

HP = TypeVar('HP', bound=HyperParameters)

class NNBase(nn.Module):

    @classmethod
    def from_hyper_parameters(cls, hyper_parameters: HP):
        return cls(**hyper_parameters.to_dict())
