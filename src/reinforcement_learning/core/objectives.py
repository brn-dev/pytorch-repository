import numpy as np
import torch

from src.function_types import TorchLossFunction, TorchReductionFunction
from src.reinforcement_learning.core.buffers.actor_critic_stm_rollout_buffer import ActorCriticSTMRolloutBuffer
from src.reinforcement_learning.core.infos import InfoDict


def compute_stm_objective(
        buffer: ActorCriticSTMRolloutBuffer,
        last_obs: np.ndarray,
        stm_loss_fn: TorchLossFunction,
        stm_objective_reduction: TorchReductionFunction,
        stm_objective_weight: float,
        info: InfoDict,
) -> torch.Tensor:
    state_preds = torch.stack(buffer.state_preds)
    state_targets = torch.concat((torch.tensor(buffer.observations[1:]), torch.tensor(last_obs).unsqueeze(0)))

    stm_objective = stm_objective_reduction(stm_loss_fn(state_preds, state_targets, reduction='none'))
    weighted_stm_objective = stm_objective_weight * stm_objective

    info['stm_objective'] = stm_objective.detach().cpu()
    info['weighted_stm_objective'] = weighted_stm_objective.detach().cpu()

    return weighted_stm_objective
