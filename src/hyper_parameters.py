import abc
from typing import Any, Optional, Union

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
    def get_hps_or_str(obj: Union['HasHyperParameters', Any]) -> HyperParameters | str:
        if isinstance(obj, HasHyperParameters):
            return obj.collect_hyper_parameters()
        return str(obj)
