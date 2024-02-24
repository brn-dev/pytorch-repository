import inspect
import json
from dataclasses import dataclass, asdict
from typing import TypeVar

from torch import nn


@dataclass(init=True)
class HyperParameters:

    def __set_item__(self, key, item):
        raise TypeError('HyperParameters are frozen, cannot set values')

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __str__(self):
        return self.serialize()

    def to_dict(self):
        return asdict(self)

    def serialize(self):
        def serialize_callable(obj):
            if callable(obj):
                source = inspect.getsource(obj).strip()
                if 'lambda' in source:
                    source = 'lambda' + source.split('lambda')[-1]
                if source.endswith(','):
                    source = source[:-1]
                return source
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        serialized_hyper_parameters = json.dumps(self.to_dict(), default=serialize_callable)
        return serialized_hyper_parameters


HP = TypeVar('HP', bound=HyperParameters)

class NNBase(nn.Module):

    @staticmethod
    def is_dropout_active(dropout_p: float | None):
        return dropout_p is not None and dropout_p > 0

    @classmethod
    def from_hyper_parameters(cls, hyper_parameters: HP):

        cls_init_params = set(inspect.signature(cls.__init__).parameters.keys()) - {'self'}

        return cls(**{
            param: hyper_parameters[param]
            for param
            in cls_init_params
        })
