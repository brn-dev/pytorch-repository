import abc
from typing import Any, Optional, Union

HyperParameters = dict[str, Any]

class HasHyperParameters(abc.ABC):

    def collect_hyper_parameters(self) -> HyperParameters:
        return {
            '__type': type(self).__name__
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
