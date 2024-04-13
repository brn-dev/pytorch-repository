from typing import Callable, Any

import gymnasium
import numpy as np
import torch
from overrides import override
from torch import nn, optim

from src.reinforcement_learning.a2c.a2c import A2C
from src.reinforcement_learning.core.buffers.actor_critic_stm_rollout_buffer import ActorCriticSTMRolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.normalization import NormalizationType
from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy
from src.reinforcement_learning.core.policies.actor_critic_stm_policy import ActorCriticSTMPolicy
from src.reinforcement_learning.core.state_transition_modelling import compute_stm_objective


class A2CSTM(A2C):

    policy: ActorCriticSTMPolicy
    buffer: ActorCriticSTMRolloutBuffer

    def __init__(
            self,
            env: gymnasium.Env,
            policy: ActorCriticSTMPolicy,
            policy_optimizer: optim.Optimizer | Callable[[ActorCriticPolicy], optim.Optimizer],
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            buffer_size: int,
            buffer_type=ActorCriticSTMRolloutBuffer,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            normalize_advantages: NormalizationType | None = None,
            actor_objective_weight: float = 1.0,
            critic_loss: nn.Module = nn.MSELoss(),
            critic_objective_weight: float = 1.0,
            stm_loss: nn.Module = nn.MSELoss(),
            stm_objective_weight: float = 1.0,
            callback: Callback['A2CSTM'] = Callback(),
    ):
        env = self.as_vec_env(env)

        super().__init__(
            env=env,
            policy=policy,
            policy_optimizer=policy_optimizer,
            select_action=select_action,
            buffer_size=buffer_size,
            buffer_type=buffer_type,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantages=normalize_advantages,
            actor_objective_weight=actor_objective_weight,
            critic_loss=critic_loss,
            critic_objective_weight=critic_objective_weight,
            callback=callback,
        )

        self.stm_loss = stm_loss
        self.stm_objective_weight = stm_objective_weight


    @override
    def compute_objectives(
            self,
            last_obs: np.ndarray,
            last_dones: np.ndarray,
            info: dict[str, Any],
    ) -> list[torch.Tensor]:
        a2c_objectives = super().compute_objectives(last_obs, last_dones, info)

        weighted_stm_objective = compute_stm_objective(
            buffer=self.buffer,
            last_obs=last_obs,
            stm_loss=self.stm_loss,
            stm_objective_weight=self.stm_objective_weight,
            info=info
        )

        return a2c_objectives + weighted_stm_objective
