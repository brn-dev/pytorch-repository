import abc
from typing import Any, Optional, Union, Callable

from torch import nn, optim

from src.torch_utils import get_lr
from src.utils import get_fully_qualified_class_name

HyperParameters = dict[str, Any]

class HasHyperParameters(abc.ABC):

    def collect_hyper_parameters(self) -> HyperParameters:
        return {
            '_type': type(self).__name__,
            '_type_fq': get_fully_qualified_class_name(self)
        }

    @staticmethod
    def update_hps(hyper_parameters: Optional[HyperParameters], values: dict[str, Any]) -> HyperParameters:
        if hyper_parameters is None:
            hyper_parameters = {}

        hyper_parameters.update(values)

        return hyper_parameters

    @staticmethod
    def get_hps_or_repr(obj: Union['HasHyperParameters', nn.Module]) -> HyperParameters | str:
        if isinstance(obj, HasHyperParameters):
            return obj.collect_hyper_parameters()
        return repr(obj)

    @staticmethod
    def maybe_get_optimizer_hps(optimizer: Optional[optim.Optimizer]) -> Optional[HyperParameters]:
        if optimizer is None:
            return None
        return HasHyperParameters.get_optimizer_hps(optimizer)

    @staticmethod
    def get_optimizer_hps(optimizer: optim.Optimizer) -> HyperParameters:
        hps: HyperParameters = {
            '_type': type(optimizer).__name__,
            '_type_fq': get_fully_qualified_class_name(optimizer)
        }

        if isinstance(optimizer, (optim.SGD, optim.Adam, optim.AdamW)):
            hps['lr'] = get_lr(optimizer)
            hps['weight_decay'] = optimizer.defaults['weight_decay']
            hps['maximize'] = optimizer.defaults['maximize']
            hps['foreach'] = optimizer.defaults['foreach']
            hps['differentiable'] = optimizer.defaults['differentiable']
            hps['fused'] = optimizer.defaults['fused']

            if isinstance(optimizer, optim.SGD):
                hps['momentum'] = optimizer.defaults['momentum']
                hps['dampening'] = optimizer.defaults['dampening']
                hps['nesterov'] = optimizer.defaults['nesterov']

            if isinstance(optimizer, (optim.Adam, optim.AdamW)):
                hps['betas'] = optimizer.defaults['betas']
                hps['eps'] = optimizer.defaults['eps']
                hps['amsgrad'] = optimizer.defaults['amsgrad']
                hps['capturable'] = optimizer.defaults['capturable']
        else:
            print(f'Warning: Optimizer type {type(optimizer)} is not set up for hyper-parameter extraction!')

        hps['repr'] = repr(optimizer)
        return hps



