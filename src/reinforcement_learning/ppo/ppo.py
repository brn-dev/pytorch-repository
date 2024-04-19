from typing import Callable, Any

import gymnasium
import numpy as np
import torch
from overrides import override
from torch import nn, optim
import torch.distributions as dist

from src.function_types import TorchReductionFunction, TorchLossFunction
from src.reinforcement_learning.core.batching import batched
from src.reinforcement_learning.core.buffers.actor_critic_rollout_buffer import ActorCriticRolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.infos import InfoDict, stack_infos, concat_infos
from src.reinforcement_learning.core.rl_base import RLBase
from src.reinforcement_learning.core.normalization import NormalizationType
from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy


class PPO(RLBase):

    policy: ActorCriticPolicy
    buffer: ActorCriticRolloutBuffer


    def __init__(
            self,
            env: gymnasium.Env,
            policy: Callable[[], ActorCriticPolicy],
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
            ppo_epochs: int = 3,
            ppo_batch_size: int | None = None,
            action_ratio_clip_range: float = 0.2,
            value_function_clip_range: float | None = None,  # Depends on return scaling
            log_unreduced: bool = False,
            callback: Callback['PPO'] = Callback(),
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

        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size if ppo_batch_size is not None else self.buffer.buffer_size

        self.action_ratio_clip_range = action_ratio_clip_range
        self.value_function_clip_range = value_function_clip_range

        self.log_unreduced = log_unreduced

    @override
    def perform_rollout(self, max_steps: int, info: InfoDict) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            return super().perform_rollout(max_steps, info)


    @override
    def optimize(
            self,
            last_obs: np.ndarray,
            last_dones: np.ndarray,
            info: InfoDict
    ) -> None:
        last_values = self.policy.predict_values(last_obs)

        advantages, returns = self.buffer.compute_gae_and_returns(
            last_values=last_values,
            last_dones=last_dones,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize_rewards=self.normalize_rewards,
            normalize_advantages=self.normalize_advantages
        )
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        info['advantages'] = advantages
        info['returns'] = returns

        observations = torch.tensor(self.buffer.observations, dtype=torch.float32)

        old_actions = torch.stack(self.buffer.actions).detach()
        old_action_log_probs = torch.stack(self.buffer.action_log_probs).detach()
        old_value_estimates = torch.stack(self.buffer.value_estimates).detach()

        optimization_infos: list[InfoDict] = []
        for _ in range(self.ppo_epochs):
            epoch_infos: list[InfoDict] = []

            for batched_tensors in batched(
                    self.ppo_batch_size,
                    observations[:self.buffer.pos], advantages[:self.buffer.pos], returns[:self.buffer.pos],
                    old_actions, old_action_log_probs, old_value_estimates
            ):
                batch_info: InfoDict = {}
                objectives = self.compute_ppo_objectives(
                    observations=batched_tensors[0],
                    advantages=batched_tensors[1],
                    returns=batched_tensors[2],
                    old_actions=batched_tensors[3],
                    old_action_log_probs=batched_tensors[4],
                    old_value_estimates=batched_tensors[5],
                    info=batch_info,
                )

                objective = torch.stack(objectives).sum()

                batch_info['objective'] = objective

                self.policy_optimizer.zero_grad()
                objective.backward()
                self.policy_optimizer.step()

                epoch_infos.append(batch_info)

            optimization_infos.append(concat_infos(epoch_infos))

        for info_key, info_value in stack_infos(optimization_infos).items():
            info[info_key] = info_value


    def compute_ppo_objectives(
            self,
            observations: torch.Tensor,
            advantages: torch.Tensor,
            returns: torch.Tensor,
            old_actions: torch.Tensor,
            old_action_log_probs: torch.Tensor,
            old_value_estimates: torch.Tensor,
            info: InfoDict,
    ) -> list[torch.Tensor]:
        new_action_logits, value_estimates = self.policy.predict_actions_and_values(observations)
        new_actions_dist = self.policy.create_actions_dist(new_action_logits)
        value_estimates = value_estimates.squeeze(dim=-1)

        actor_objective = self.compute_ppo_actor_objective(
            new_actions_dist=new_actions_dist,
            advantages=advantages,
            old_actions=old_actions,
            old_action_log_probs=old_action_log_probs,
            action_ratio_clip_range=self.action_ratio_clip_range,
            actor_objective_reduction=self.actor_objective_reduction,
            actor_objective_weight=self.actor_objective_weight,
            info=info,
            log_unreduced=self.log_unreduced,
        )

        critic_objective = self.compute_ppo_critic_objective(
            returns=returns,
            old_value_estimates=old_value_estimates,
            new_value_estimates=value_estimates,
            value_function_clip_range=self.value_function_clip_range,
            critic_loss_fn=self.critic_loss_fn,
            critic_objective_reduction=self.critic_objective_reduction,
            critic_objective_weight=self.critic_objective_weight,
            info=info,
            log_unreduced=self.log_unreduced,
        )

        return [actor_objective, critic_objective]

    @staticmethod
    def compute_ppo_actor_objective(
            new_actions_dist: dist.Distribution,
            advantages: torch.Tensor,
            old_actions: torch.Tensor,
            old_action_log_probs: torch.Tensor,
            action_ratio_clip_range,
            actor_objective_reduction: TorchReductionFunction,
            actor_objective_weight: float,
            info: InfoDict,
            log_unreduced: bool
    ) -> torch.Tensor:
        new_action_log_probs = new_actions_dist.log_prob(old_actions)

        action_log_probs_ratios = torch.exp(new_action_log_probs - old_action_log_probs)

        unclipped_actor_objective = advantages * action_log_probs_ratios
        clipped_actor_objective = advantages * torch.clamp(
            action_log_probs_ratios,
            1 - action_ratio_clip_range,
            1 + action_ratio_clip_range
        )
        actor_objective_unreduced = -torch.min(unclipped_actor_objective, clipped_actor_objective)
        actor_objective = actor_objective_reduction(actor_objective_unreduced)
        weighted_actor_objective = actor_objective_weight * actor_objective

        info['actor_objective'] = actor_objective.detach().cpu()
        info['weighted_actor_objective'] = weighted_actor_objective.detach().cpu()

        if log_unreduced:
            info['actor_objective_unreduced'] = actor_objective_unreduced.detach().cpu()

        return weighted_actor_objective

    @staticmethod
    def compute_ppo_critic_objective(
            returns: torch.Tensor,
            old_value_estimates: torch.Tensor,
            new_value_estimates: torch.Tensor,
            value_function_clip_range: float | None,
            critic_loss_fn: TorchLossFunction,
            critic_objective_reduction: TorchReductionFunction,
            critic_objective_weight: float,
            info: InfoDict,
            log_unreduced: bool,
    ) -> torch.Tensor:
        value_estimates = new_value_estimates

        if value_function_clip_range is not None:
            value_estimates = old_value_estimates + torch.clamp(
                new_value_estimates - old_value_estimates, -value_function_clip_range, value_function_clip_range
            )

        critic_objective_unreduced = critic_loss_fn(value_estimates, returns, reduction='none')
        critic_objective = critic_objective_reduction(critic_objective_unreduced)
        weighted_critic_objective = critic_objective_weight * critic_objective

        info['critic_objective'] = critic_objective.detach().cpu()
        info['weighted_critic_objective'] = weighted_critic_objective.detach().cpu()

        if log_unreduced:
            info['critic_objective_unreduced'] = critic_objective_unreduced.detach().cpu()

        return weighted_critic_objective
