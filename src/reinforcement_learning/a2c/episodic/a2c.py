from typing import Callable, Any

import gymnasium
import numpy as np
import torch
from torch import nn, optim

from src.reinforcement_learning.core.buffers.actor_critic_rollout_buffer import ActorCriticRolloutBuffer
from src.reinforcement_learning.core.episodic_rl_base import EpisodicRLBase, RolloutDoneCallback, \
    OptimizationDoneCallback
from src.reinforcement_learning.core.normalization import NormalizationType
from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy


class A2C(EpisodicRLBase):

    policy: ActorCriticPolicy
    buffer: ActorCriticRolloutBuffer


    def __init__(
            self,
            env: gymnasium.Env,
            policy: ActorCriticPolicy,
            policy_optimizer: optim.Optimizer | Callable[[ActorCriticPolicy], optim.Optimizer],
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            buffer_size: int,
            gamma: float,
            gae_lambda: float,
            normalize_advantages: NormalizationType,
            critic_loss: nn.Module,
            critic_objective_weight: float,
            on_rollout_done: RolloutDoneCallback,
            on_optimization_done: OptimizationDoneCallback,
            buffer_type=ActorCriticRolloutBuffer,
    ):
        env = self.as_vec_env(env)

        super().__init__(
            env=env,
            policy=policy,
            select_action=select_action,
            buffer=buffer_type(buffer_size, env.num_envs, env.observation_space.shape),
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantages=normalize_advantages,
            on_rollout_done=on_rollout_done,
            on_optimization_done=on_optimization_done,
        )

        self.policy_optimizer = (
            policy_optimizer
            if isinstance(policy_optimizer, optim.Optimizer)
            else policy_optimizer(policy)
        )

        self.critic_loss = critic_loss
        self.critic_objective_weight = critic_objective_weight

    def optimize(self, last_obs: np.ndarray, last_dones: np.ndarray) -> None:
        last_values = self.policy.predict_values(last_obs)

        advantages, returns = self.compute_gae_and_returns(last_values, last_dones)
        advantages = torch.tensor(advantages)
        returns = torch.tensor(returns)

        action_log_probs = torch.stack(self.buffer.action_log_probs)
        value_estimates = torch.stack(self.buffer.value_estimates)

        actor_objective = -(action_log_probs * advantages).mean()
        critic_objective = self.critic_loss(value_estimates, returns)

        combined_objective = actor_objective + self.critic_objective_weight * critic_objective

        self.policy_optimizer.zero_grad()
        combined_objective.backward()
        self.policy_optimizer.step()
