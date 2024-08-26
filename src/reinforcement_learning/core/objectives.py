from dataclasses import dataclass

import torch

from src.function_types import TorchTensorTransformation
from src.reinforcement_learning.core.infos import InfoDict


@dataclass
class ObjectiveLoggingConfig:
    log_raw: bool = False
    log_reduced: bool = False


def weigh_and_reduce_objective(
        raw_objective: torch.Tensor,
        weigh_and_reduce_function: TorchTensorTransformation,
        info: InfoDict,
        objective_name: str,
        logging_config: ObjectiveLoggingConfig
):
    reduced_objective = weigh_and_reduce_function(raw_objective)

    if logging_config.log_raw:
        info[f'raw_{objective_name}'] = raw_objective.detach()
    if logging_config.log_reduced:
        info[f'reduced_{objective_name}'] = reduced_objective.detach()

    return reduced_objective


