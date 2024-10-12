from dataclasses import dataclass

import torch

from src.function_types import TorchTensorFn
from src.reinforcement_learning.core.infos import InfoDict


@dataclass
class LossInfoStashConfig:
    stash_raw: bool = False
    stash_final: bool = False

    def __post_init__(self):
        self.stash_anything = self.stash_raw or self.stash_final


# TODO: add flag to put log tensors on cpu right here (otherwise will be done in concat_infos)
# TODO: The current approach is more time efficient but needs a bit more vram
def weigh_and_reduce_loss(
        raw_loss: torch.Tensor,
        weigh_and_reduce_function: TorchTensorFn,
        info: InfoDict,
        loss_name: str,
        stash_config: LossInfoStashConfig
):
    reduced_loss = weigh_and_reduce_function(raw_loss)

    if stash_config.stash_raw:
        info[f'raw_{loss_name}'] = raw_loss.detach()
    if stash_config.stash_final:
        info[f'final_{loss_name}'] = reduced_loss.detach()

    return reduced_loss


