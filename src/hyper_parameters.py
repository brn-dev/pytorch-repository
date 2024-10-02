import abc
from typing import Any, Optional, Union, Callable

from torch import nn, optim

from src.repr_utils import func_repr
from src.torch_nn_modules import is_nn_activation_module
from src.torch_utils import get_lr
from src.utils import get_fully_qualified_class_name

HyperParameters = dict[str, Any]

TYPE_KEY = '_type'
FQ_TYPE_KEY = '_type_fq'


class HasHyperParameters(abc.ABC):

    def collect_hyper_parameters(self) -> HyperParameters:
        return HasHyperParameters.get_type_hps(self)

    #
    # making static methods instead of function so one doesn't have to import them manually
    # TODO: pull out as separate functions and alias
    #

    @staticmethod
    def get_type_hps(obj: Any) -> HyperParameters:
        return {
            TYPE_KEY: type(obj).__name__,
            FQ_TYPE_KEY: get_fully_qualified_class_name(obj)
        }

    @staticmethod
    def update_hps(hyper_parameters: Optional[HyperParameters], values: dict[str, Any]) -> HyperParameters:
        if hyper_parameters is None:
            hyper_parameters = {}

        hyper_parameters.update(values)

        return hyper_parameters

    @staticmethod
    def maybe_get_func_repr(f: Optional[Callable]) -> Optional[str]:
        if f is None:
            return None

        return func_repr(f)

    @staticmethod
    def get_func_repr(f: Callable) -> str:
        return func_repr(f)

    @staticmethod
    def maybe_get_optimizer_hps(optimizer: Optional[optim.Optimizer]) -> Optional[HyperParameters]:
        if optimizer is None:
            return None
        return HasHyperParameters.get_optimizer_hps(optimizer)

    @staticmethod
    def get_optimizer_hps(optimizer: optim.Optimizer) -> HyperParameters:
        hps: HyperParameters = HasHyperParameters.get_type_hps(optimizer)

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

    @staticmethod
    def get_module_hps(module: Union['HasHyperParameters', nn.Module], add_repr: bool = True) -> HyperParameters:
        hps = HasHyperParameters.get_type_hps(module)
        known_type = True
        if isinstance(module, HasHyperParameters):
            hps = HasHyperParameters.update_hps(hps, module.collect_hyper_parameters())
        elif isinstance(module, nn.Linear):
            hps = HasHyperParameters.update_hps(hps, {
                'in_features': module.in_features,
                'out_features': module.out_features,
                'bias': module.bias is not None,
            })
        elif isinstance(module, nn.Sequential):
            hps = HasHyperParameters.update_hps(hps, {
                'num_layers': len(module),
                'layers': [
                    HasHyperParameters.get_module_hps(layer, add_repr=False)
                    for layer in module
                ]
            })
        else:
            known_type = False

        if not known_type or add_repr:
            hps['repr'] = repr(module)

        return hps




