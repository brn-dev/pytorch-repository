import abc
from typing import Callable

from torch import nn

class NNBase(abc.ABC, nn.Module):

    @staticmethod
    def is_dropout_active(dropout_p: float | None):
        return dropout_p is not None and dropout_p > 0

    @staticmethod
    def create_dropout(dropout_p: float | None):
        if NNBase.is_dropout_active(dropout_p):
            return nn.Dropout(dropout_p)
        return None

    @staticmethod
    def provide(provider: Callable[..., nn.Module] | None, *args, **kwargs):
        if provider is None:
            return None
        return provider(*args, **kwargs)
