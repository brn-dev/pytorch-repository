from dataclasses import dataclass

import torch

from src.function_types import TorchTensorFn
from src.reinforcement_learning.core.infos import InfoDict


@dataclass
class LossLoggingConfig:
    log_raw: bool = False
    log_final: bool = False


# TODO: add flag to put log tensors on cpu right here (otherwise will be done in concat_infos)
# TODO: The current approach is more time efficient but needs a bit more vram
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
    if logging_config.log_final:
        info[f'final_{loss_name}'] = reduced_loss.detach()

    return reduced_loss


