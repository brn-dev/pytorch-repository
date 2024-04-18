from typing import Callable, Any

import gymnasium
import numpy as np
import torch
from overrides import override
from torch import nn, optim

from src.function_types import TorchReductionFunction, TorchLossFunction
from src.reinforcement_learning.core.buffers.actor_critic_rollout_buffer import ActorCriticRolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.rl_base import RLBase
from src.reinforcement_learning.core.normalization import NormalizationType
from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy


class A2C(RLBase):

    policy: ActorCriticPolicy
    buffer: ActorCriticRolloutBuffer


    def __init__(
            self,
            env: gymnasium.Env,
            policy: ActorCriticPolicy,
            policy_optimizer: optim.Optimizer | Callable[[ActorCriticPolicy], optim.Optimizer],
            buffer_size: int,
            buffer_type=ActorCriticRolloutBuffer,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            normalize_rewards: NormalizationType | None = None,
            normalize_advantages: NormalizationType | None = None,
            actor_objective_reduction: TorchReductionFunction = torch.mean,
            actor_objective_weight: float = 1.0,
            critic_loss_fn: TorchLossFunction = nn.functional.mse_loss,
            critic_objective_reduction: TorchReductionFunction = torch.mean,
            critic_objective_weight: float = 1.0,
            log_unreduced: bool = False,
            callback: Callback['A2C'] = Callback(),
    ):
        env = self.as_vec_env(env)

        super().__init__(
            env=env,
            policy=policy,
            policy_optimizer=policy_optimizer,
            buffer=buffer_type(buffer_size, env.num_envs, env.observation_space.shape),
            gamma=gamma,
            gae_lambda=gae_lambda,
            callback=callback
        )

        self.normalize_rewards = normalize_rewards
        self.normalize_advantages = normalize_advantages

        self.actor_objective_reduction = actor_objective_reduction
        self.actor_objective_weight = actor_objective_weight

        self.critic_loss_fn = critic_loss_fn
        self.critic_objective_reduction = critic_objective_reduction
        self.critic_objective_weight = critic_objective_weight

        self.log_unreduced = log_unreduced


    @override
    def optimize(
            self,
            last_obs: np.ndarray,
            last_dones: np.ndarray,
            info: dict[str, Any]
    ) -> None:
        objectives = self.compute_objectives(last_obs, last_dones, info)

        objective = torch.stack(objectives).sum()

        info['objective'] = objective

        self.policy_optimizer.zero_grad()
        objective.backward()
        self.policy_optimizer.step()

    def compute_objectives(
            self,
            last_obs: np.ndarray,
            last_dones: np.ndarray,
            info: dict[str, Any],
    ) -> list[torch.Tensor]:
        actor_objective, critic_objective = self.compute_a2c_objectives(
            last_obs=last_obs,
            last_dones=last_dones,
            info=info,
            actor_objective_reduction=self.actor_objective_reduction,
            actor_objective_weight=self.actor_objective_weight,
            critic_loss_fn=self.critic_loss_fn,
            critic_objective_reduction=self.critic_objective_reduction,
            critic_objective_weight=self.critic_objective_weight,
        )

        return [actor_objective, critic_objective]

    def compute_a2c_objectives(
            self,
            last_obs: np.ndarray,
            last_dones: np.ndarray,
            info: dict[str, Any],
            actor_objective_reduction: TorchReductionFunction,
            actor_objective_weight: float,
            critic_loss_fn: TorchLossFunction,
            critic_objective_reduction: TorchReductionFunction,
            critic_objective_weight,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        last_values = self.policy.predict_values(last_obs)

        advantages_np, returns_np = self.buffer.compute_gae_and_returns(
            last_values=last_values,
            last_dones=last_dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize_rewards=self.normalize_rewards,
            normalize_advantages=self.normalize_advantages,
        )
        advantages = torch.tensor(advantages_np, dtype=torch.float32)
        returns = torch.tensor(returns_np, dtype=torch.float32)

        action_log_probs = (torch.stack(self.buffer.action_log_probs)
                            .reshape((self.buffer.pos, self.env.num_envs, -1))
                            .mean(dim=-1))
        value_estimates = torch.stack(self.buffer.value_estimates)

        actor_objective_unreduced = -action_log_probs * advantages
        critic_objective_unreduced = critic_loss_fn(value_estimates, returns, reduction='none')

        actor_objective = actor_objective_reduction(actor_objective_unreduced)
        critic_objective = critic_objective_reduction(critic_objective_unreduced)

        weighted_actor_objective = actor_objective_weight * actor_objective
        weighted_critic_objective = critic_objective_weight * critic_objective

        info['advantages'] = advantages
        info['returns'] = returns
        info['actor_objective'] = actor_objective.detach().cpu()
        info['weighted_actor_objective'] = weighted_actor_objective.detach().cpu()
        info['critic_objective'] = critic_objective.detach().cpu()
        info['weighted_critic_objective'] = weighted_critic_objective.detach().cpu()

        if self.log_unreduced:
            info['actor_objective_unreduced'] = actor_objective_unreduced.detach().cpu()
            info['critic_objective_unreduced'] = critic_objective_unreduced.detach().cpu()

        return weighted_actor_objective, weighted_critic_objective

