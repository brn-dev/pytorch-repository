from dataclasses import dataclass

import torch

from src.function_types import TorchReductionFunction, TorchTensorTransformation
from src.reinforcement_learning.core.infos import InfoDict


@dataclass
class ObjectiveLoggingConfig:
    log_raw: bool = False
    log_reduced: bool = False
    log_weighted: bool = False


def reduce_and_weigh_objective(
        raw_objective: torch.Tensor,
        reduce_objective: TorchReductionFunction,
        weigh_objective: TorchTensorTransformation,
        info: InfoDict,
        objective_name: str,
        logging_config: ObjectiveLoggingConfig
):
    reduced_objective = reduce_objective(raw_objective)

    weighted_critic_objective = weigh_objective(reduced_objective)

    if logging_config.log_raw:
        info[f'raw_{objective_name}'] = raw_objective.detach().cpu()
    if logging_config.log_reduced:
        info[f'reduced_{objective_name}'] = reduced_objective.detach().cpu()
    if logging_config.log_weighted:
        info[f'weighted_{objective_name}'] = weighted_critic_objective.detach().cpu()

    return weighted_critic_objective
