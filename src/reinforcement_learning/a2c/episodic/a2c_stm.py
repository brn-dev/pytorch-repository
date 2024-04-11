from typing import Callable, Any

import gymnasium
import numpy as np
import torch
from torch import nn, optim

from src.reinforcement_learning.a2c.episodic.a2c import A2C
from src.reinforcement_learning.core.buffers.actor_critic_stm_rollout_buffer import ActorCriticSTMRolloutBuffer
from src.reinforcement_learning.core.episodic_rl_base import RolloutDoneCallback, \
    OptimizationDoneCallback
from src.reinforcement_learning.core.normalization import NormalizationType
from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy
from src.reinforcement_learning.core.policies.actor_critic_stm_policy import ActorCriticSTMPolicy


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
            gamma: float,
            gae_lambda: float,
            normalize_advantages: NormalizationType,
            actor_objective_weight: float,
            critic_loss: nn.Module,
            critic_objective_weight: float,
            stm_loss: nn.Module,
            stm_objective_weight: float,
            on_rollout_done: RolloutDoneCallback,
            on_optimization_done: OptimizationDoneCallback,
            buffer_type=ActorCriticSTMRolloutBuffer,
    ):
        env = self.as_vec_env(env)

        super().__init__(
            env=env,
            policy=policy,
            policy_optimizer=policy_optimizer,
            select_action=select_action,
            buffer_size=buffer_size,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantages=normalize_advantages,
            critic_loss=critic_loss,
            critic_objective_weight=critic_objective_weight,
            on_rollout_done=on_rollout_done,
            on_optimization_done=on_optimization_done,
            buffer_type=buffer_type,
        )

        self.actor_objective_weight = actor_objective_weight
        self.stm_loss = stm_loss
        self.stm_objective_weight = stm_objective_weight


    def optimize(self, last_obs: np.ndarray, last_dones: np.ndarray) -> None:
        last_values = self.policy.predict_values(last_obs)

        advantages, returns = self.compute_gae_and_returns(last_values, last_dones)
        advantages = torch.tensor(advantages)
        returns = torch.tensor(returns)

        action_log_probs = torch.stack(self.buffer.action_log_probs)
        value_estimates = torch.stack(self.buffer.value_estimates)
        state_preds = torch.stack(self.buffer.state_preds)
        state_targets = torch.stack([torch.tensor(self.buffer.observations[1:]), torch.tensor(last_obs).unsqueeze(0)])

        actor_objective = -(action_log_probs * advantages).mean()
        critic_objective = self.critic_loss(value_estimates, returns)
        stm_objective = self.stm_loss(state_preds, state_targets)

        combined_objective = (
                self.actor_objective_weight * actor_objective
                + self.critic_objective_weight * critic_objective
                + self.stm_objective_weight * stm_objective
        )

        self.policy_optimizer.zero_grad()
        combined_objective.backward()
        self.policy_optimizer.step()
