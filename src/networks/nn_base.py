import abc
from typing import Callable

from torch import nn


class NNBase(abc.ABC, nn.Module):

    Provider = Callable[..., nn.Module] | nn.Module | None

    @staticmethod
    def provide(provider: Provider, *args, _if: bool = True, **kwargs):
        if not _if or provider is None:
            return None
        if isinstance(provider, Callable):
            module = provider(*args, **kwargs)

            if not isinstance(module, nn.Module):
                raise ValueError(f'Provider did not return a module ({module = })')

            return module
        raise ValueError(f'Unknown Provider type ({provider = })')

    @staticmethod
    def is_dropout_active(dropout_p: float | None):
        return dropout_p is not None and dropout_p > 0

    @staticmethod
    def provide_dropout(dropout_p: float | None, _if: bool = True):
        return NNBase.provide(nn.Dropout(dropout_p), _if=_if and NNBase.is_dropout_active(dropout_p))
