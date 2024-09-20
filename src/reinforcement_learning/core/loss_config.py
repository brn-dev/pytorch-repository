from dataclasses import dataclass

import torch

from src.function_types import TorchTensorFn
from src.reinforcement_learning.core.infos import InfoDict


@dataclass
class LossLoggingConfig:
    log_raw: bool = False
    log_reduced: bool = False


def weigh_and_reduce_loss(
        raw_loss: torch.Tensor,
        weigh_and_reduce_function: TorchTensorFn,
        info: InfoDict,
        loss_name: str,
        logging_config: LossLoggingConfig
):
    reduced_loss = weigh_and_reduce_function(raw_loss)

    if logging_config.log_raw:
        info[f'raw_{loss_name}'] = raw_loss.detach()
    if logging_config.log_reduced:
        info[f'reduced_{loss_name}'] = reduced_loss.detach()

    return reduced_loss


