from dataclasses import dataclass
from typing import Callable, Any, Optional

import gymnasium
import numpy as np
import torch
from overrides import override
from torch import nn, optim
import torch.distributions as dist

from src.function_types import TorchReductionFunction, TorchLossFunction, TorchTensorTransformation
from src.reinforcement_learning.core.batching import batched
from src.reinforcement_learning.core.buffers.actor_critic_rollout_buffer import ActorCriticRolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.infos import InfoDict, stack_infos, concat_infos
from src.reinforcement_learning.core.objectives import reduce_and_weigh_objective, ObjectiveLoggingConfig
from src.reinforcement_learning.core.rl_base import RLBase, LoggingConfig
from src.reinforcement_learning.core.normalization import NormalizationType
from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy


@dataclass
class PPOLoggingConfig(LoggingConfig):
    log_returns: bool = False
    log_advantages: bool = False

    log_ppo_objective: bool = False
    log_actor_kl_divergence: bool = False
    actor_objective: ObjectiveLoggingConfig = None
    critic_objective: ObjectiveLoggingConfig = None

    def __post_init__(self):
        if self.actor_objective is None:
            self.actor_objective = ObjectiveLoggingConfig()
        if self.critic_objective is None:
            self.critic_objective = ObjectiveLoggingConfig()


class PPO(RLBase):

    policy: ActorCriticPolicy
    buffer: ActorCriticRolloutBuffer
    logging_config: PPOLoggingConfig

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
            reduce_actor_objective: TorchReductionFunction = torch.mean,
            weigh_actor_objective: TorchTensorTransformation = lambda obj: obj,
            critic_loss_fn: TorchLossFunction = nn.functional.mse_loss,
            reduce_critic_objective: TorchReductionFunction = torch.mean,
            weigh_critic_objective: TorchTensorTransformation = lambda obj: obj,
            ppo_max_epochs: int = 5,
            ppo_kl_target: float | None = None,
            ppo_batch_size: int | None = None,
            action_ratio_clip_range: float = 0.2,
            value_function_clip_range_factor: float | None = None,
            reset_env_between_rollouts: bool = False,
            callback: Callback['PPO'] = None,
            logging_config: PPOLoggingConfig = None,
    ):
        env, num_envs = self.as_vec_env(env)

        super().__init__(
            env=env,
            policy=policy,
            policy_optimizer=policy_optimizer,
            buffer=buffer_type(buffer_size, num_envs, env.observation_space.shape),
            gamma=gamma,
            gae_lambda=gae_lambda,
            reset_env_between_rollouts=reset_env_between_rollouts,
            callback=callback or Callback(),
            logging_config=logging_config or PPOLoggingConfig()
        )

        self.normalize_rewards = normalize_rewards
        self.normalize_advantages = normalize_advantages

        self.reduce_actor_objective = reduce_actor_objective
        self.weigh_actor_objective = weigh_actor_objective

        self.critic_loss_fn = critic_loss_fn
        self.reduce_critic_objective = reduce_critic_objective
        self.weigh_critic_objective = weigh_critic_objective

        self.ppo_max_epochs = ppo_max_epochs
        self.ppo_kl_target = ppo_kl_target
        self.ppo_batch_size = ppo_batch_size if ppo_batch_size is not None else self.buffer.buffer_size

        self.action_ratio_clip_range = action_ratio_clip_range
        self.value_function_clip_range_factor = value_function_clip_range_factor

    @override
    def perform_rollout(
            self,
            max_steps: int,
            obs: np.ndarray,
            info: InfoDict
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        with torch.no_grad():
            return super().perform_rollout(max_steps, obs, info)


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

        if self.logging_config.log_advantages:
            info['advantages'] = advantages
        if self.logging_config.log_returns:
            info['returns'] = returns

        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        observations = torch.tensor(self.buffer.observations, dtype=torch.float32)

        old_actions = torch.stack(self.buffer.actions).detach()
        old_action_log_probs = torch.stack(self.buffer.action_log_probs).detach()
        old_value_estimates = torch.stack(self.buffer.value_estimates).detach()

        optimization_infos: list[InfoDict] = []
        continue_training = True
        nr_updates = 0
        i_epoch = 0
        for i_epoch in range(self.ppo_max_epochs):
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

                if objectives is None:
                    continue_training = False
                    epoch_infos.append(batch_info)
                    break

                objective = torch.stack(objectives).sum()

                if self.logging_config.log_ppo_objective:
                    batch_info['objective'] = objective

                self.policy_optimizer.zero_grad()
                objective.backward()
                self.policy_optimizer.step()

                epoch_infos.append(batch_info)
                nr_updates += 1

            optimization_infos.append(concat_infos(epoch_infos))

            if not continue_training:
                break

        for info_key, info_value in concat_infos(optimization_infos).items():
            info[info_key] = info_value

        info['nr_ppo_epochs'] = i_epoch + 1
        info['nr_ppo_updates'] = nr_updates


    def compute_ppo_objectives(
            self,
            observations: torch.Tensor,
            advantages: torch.Tensor,
            returns: torch.Tensor,
            old_actions: torch.Tensor,
            old_action_log_probs: torch.Tensor,
            old_value_estimates: torch.Tensor,
            info: InfoDict,
    ) -> Optional[list[torch.Tensor]]:
        new_action_logits, value_estimates = self.policy.predict_actions_and_values(observations)
        new_actions_dist = self.policy.action_dist_provider(new_action_logits)

        value_estimates = value_estimates.squeeze(dim=-1)

        actor_objective = self.compute_ppo_actor_objective(
            new_actions_dist=new_actions_dist,
            advantages=advantages,
            old_actions=old_actions,
            old_action_log_probs=old_action_log_probs,
            info=info,
        )

        if actor_objective is None:
            return None

        critic_objective = self.compute_ppo_critic_objective(
            returns=returns,
            old_value_estimates=old_value_estimates,
            new_value_estimates=value_estimates,
            info=info,
        )

        return [actor_objective, critic_objective]

    def compute_ppo_actor_objective(
            self,
            new_actions_dist: dist.Distribution,
            advantages: torch.Tensor,
            old_actions: torch.Tensor,
            old_action_log_probs: torch.Tensor,
            info: InfoDict,
    ) -> Optional[torch.Tensor]:
        new_action_log_probs = new_actions_dist.log_prob(old_actions)

        if self.ppo_kl_target is not None or self.logging_config.log_actor_kl_divergence:
            # https://github.com/DLR-RM/stable-baselines3/blob/285e01f64aa8ba4bd15aa339c45876d56ed0c3b4/stable_baselines3/ppo/ppo.py#L266
            with torch.no_grad():
                log_ratio = new_action_log_probs - old_action_log_probs
                approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().unsqueeze(0).numpy()

            if self.logging_config.log_actor_kl_divergence:
                info['actor_kl_divergence'] = approx_kl_div

            if self.ppo_kl_target is not None and approx_kl_div > 1.5 * self.ppo_kl_target:
                return None

        action_log_probs_ratios = torch.exp(new_action_log_probs - old_action_log_probs)

        if action_log_probs_ratios.dim() > 2:
            action_log_probs_ratios = action_log_probs_ratios.flatten(start_dim=2).sum(dim=2)

        unclipped_actor_objective = advantages * action_log_probs_ratios
        clipped_actor_objective = advantages * torch.clamp(
            action_log_probs_ratios,
            1 - self.action_ratio_clip_range,
            1 + self.action_ratio_clip_range
        )
        raw_actor_objective = -torch.min(unclipped_actor_objective, clipped_actor_objective)

        actor_objective = reduce_and_weigh_objective(
            raw_objective=raw_actor_objective,
            reduce_objective=self.reduce_actor_objective,
            weigh_objective=self.weigh_actor_objective,
            info=info,
            objective_name='actor_objective',
            logging_config=self.logging_config.actor_objective,
        )

        return actor_objective

    def compute_ppo_critic_objective(
            self,
            returns: torch.Tensor,
            old_value_estimates: torch.Tensor,
            new_value_estimates: torch.Tensor,
            info: InfoDict,
    ) -> torch.Tensor:
        value_estimates = new_value_estimates

        if self.value_function_clip_range_factor is not None:
            with (torch.no_grad()):
                clip_range = self.value_function_clip_range_factor * torch.abs(old_value_estimates)

            value_estimates = old_value_estimates + torch.clamp(
                new_value_estimates - old_value_estimates, -clip_range, clip_range
            )

        raw_critic_objective = self.critic_loss_fn(value_estimates, returns, reduction='none')

        critic_objective = reduce_and_weigh_objective(
            raw_objective=raw_critic_objective,
            reduce_objective=self.reduce_critic_objective,
            weigh_objective=self.weigh_critic_objective,
            info=info,
            objective_name='critic_objective',
            logging_config=self.logging_config.critic_objective
        )

        return critic_objective
