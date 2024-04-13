from typing import Callable, Any

import gymnasium
import numpy as np
import torch
from overrides import override
from torch import nn, optim

from src.reinforcement_learning.core.buffers.actor_critic_rollout_buffer import ActorCriticRolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.episodic_rl_base import EpisodicRLBase
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
            buffer_type=ActorCriticRolloutBuffer,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            normalize_advantages: NormalizationType | None = None,
            actor_objective_weight: float = 1.0,
            critic_loss: nn.Module = nn.MSELoss(),
            critic_objective_weight: float = 1.0,
            callback: Callback['A2C'] = Callback(),
    ):
        env = self.as_vec_env(env)

        super().__init__(
            env=env,
            policy=policy,
            policy_optimizer=policy_optimizer,
            select_action=select_action,
            buffer=buffer_type(buffer_size, env.num_envs, env.observation_space.shape),
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_advantages=normalize_advantages,
            callback=callback
        )

        self.actor_objective_weight = actor_objective_weight

        self.critic_loss = critic_loss
        self.critic_objective_weight = critic_objective_weight

    @override
    def compute_objectives(
            self,
            last_obs: np.ndarray,
            last_dones: np.ndarray,
            info: dict[str, Any],
    ) -> list[torch.Tensor]:
        actor_objective, critic_objective, advantages, returns = self.compute_a2c_objectives(last_obs, last_dones)

        weighted_actor_objective = self.actor_objective_weight * actor_objective
        weighted_critic_objective = self.critic_objective_weight * critic_objective

        info['advantages'] = advantages
        info['returns'] = returns
        info['actor_objective'] = actor_objective.detach().cpu()
        info['weighted_actor_objective'] = weighted_actor_objective.detach().cpu()
        info['critic_objective'] = critic_objective.detach().cpu()
        info['weighted_critic_objective'] = weighted_critic_objective.detach().cpu()

        return [weighted_actor_objective, weighted_critic_objective]

    def compute_a2c_objectives(
            self,
            last_obs: np.ndarray,
            last_dones: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
        last_values = self.policy.predict_values(last_obs)

        advantages_np, returns_np = self.buffer.compute_gae_and_returns(
            last_values,
            last_dones,
            self.gamma,
            self.gae_lambda,
            self.normalize_advantages,
        )
        advantages = torch.tensor(advantages_np, dtype=torch.float32)
        returns = torch.tensor(returns_np, dtype=torch.float32)

        action_log_probs = torch.stack(self.buffer.action_log_probs)
        value_estimates = torch.stack(self.buffer.value_estimates)

        actor_objective = -(action_log_probs * advantages.unsqueeze(-1)).mean()
        critic_objective = self.critic_loss(value_estimates, returns)

        return actor_objective, critic_objective, advantages_np, returns_np

