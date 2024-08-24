from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium
import numpy as np
import torch
from overrides import override
from torch import nn, optim

from src.function_types import TorchLossFunction, TorchTensorTransformation
from src.module_analysis import calculate_grad_norm
from src.reinforcement_learning.algorithms.base.on_policy_algorithm import OnPolicyAlgorithm, LoggingConfig, \
    PolicyProvider, Policy
from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.buffers.rollout.rollout_buffer import RolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.infos import InfoDict, concat_infos
from src.reinforcement_learning.core.normalization import NormalizationType
from src.reinforcement_learning.core.objectives import weigh_and_reduce_objective, ObjectiveLoggingConfig
from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy
from src.reinforcement_learning.gym.singleton_vector_env import as_vec_env
from src.torch_device import TorchDevice
from src.type_aliases import KwArgs


@dataclass
class PPOLoggingConfig(LoggingConfig):
    log_returns: bool = False
    log_advantages: bool = False

    log_ppo_objective: bool = False
    log_grad_norm: bool = False
    log_actor_kl_divergence: bool = False
    log_optimization_action_stds: bool = False

    actor_objective: ObjectiveLoggingConfig = None
    entropy_objective: ObjectiveLoggingConfig = None
    critic_objective: ObjectiveLoggingConfig = None

    def __post_init__(self):
        if self.actor_objective is None:
            self.actor_objective = ObjectiveLoggingConfig()
        if self.entropy_objective is None:
            self.entropy_objective = ObjectiveLoggingConfig()
        if self.critic_objective is None:
            self.critic_objective = ObjectiveLoggingConfig()


class PPO(OnPolicyAlgorithm[ActorCriticPolicy, RolloutBuffer, PPOLoggingConfig]):

    def __init__(
            self,
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            policy_optimizer: optim.Optimizer | Callable[[ActorCriticPolicy], optim.Optimizer],
            buffer_size: int,
            buffer_type=RolloutBuffer,
            buffer_kwargs: KwArgs = None,
            gamma: float = 0.99,
            gae_lambda: float = 1.0,
            normalize_rewards: NormalizationType | None = None,
            normalize_advantages: NormalizationType | None = None,
            weigh_and_reduce_actor_objective: TorchTensorTransformation = torch.mean,
            weigh_and_reduce_entropy_objective: TorchTensorTransformation | None = None,
            critic_loss_fn: TorchLossFunction = nn.functional.mse_loss,
            weigh_and_reduce_critic_objective: TorchTensorTransformation = torch.mean,
            ppo_max_epochs: int = 5,
            ppo_kl_target: float | None = None,
            ppo_batch_size: int | None = None,
            shuffle_batches: bool = False,
            action_ratio_clip_range: float = 0.2,
            value_function_clip_range_factor: float | None = None,
            grad_norm_clip_value: float | None = None,
            sde_noise_sample_freq: int | None = None,
            reset_env_between_rollouts: bool = False,
            callback: Callback['PPO'] = None,
            logging_config: PPOLoggingConfig = None,
            torch_device: TorchDevice = 'cpu',
            torch_dtype: torch.dtype = torch.float32,
    ):
        env, num_envs = as_vec_env(env)

        super().__init__(
            env=env,
            policy=policy,
            policy_optimizer=policy_optimizer,
            buffer=buffer_type.for_env(env, buffer_size, torch_device, torch_dtype, **(buffer_kwargs or {})),
            gamma=gamma,
            gae_lambda=gae_lambda,
            sde_noise_sample_freq=sde_noise_sample_freq,
            reset_env_between_rollouts=reset_env_between_rollouts,
            callback=callback or Callback(),
            logging_config=logging_config or PPOLoggingConfig(),
            torch_device=torch_device,
            torch_dtype=torch_dtype,
        )

        self.normalize_rewards = normalize_rewards
        self.normalize_advantages = normalize_advantages

        self.weigh_and_reduce_actor_objective = weigh_and_reduce_actor_objective
        self.weigh_and_reduce_entropy_objective = weigh_and_reduce_entropy_objective

        self.critic_loss_fn = critic_loss_fn
        self.weigh_and_reduce_critic_objective = weigh_and_reduce_critic_objective

        self.ppo_max_epochs = ppo_max_epochs
        self.ppo_kl_target = ppo_kl_target
        self.ppo_batch_size = ppo_batch_size if ppo_batch_size is not None else self.buffer.buffer_size
        self.shuffle_batches = shuffle_batches

        self.action_ratio_clip_range = action_ratio_clip_range
        self.value_function_clip_range_factor = value_function_clip_range_factor

        self.grad_norm_clip_value = grad_norm_clip_value

    @override
    def perform_rollout(
            self,
            max_steps: int,
            obs: np.ndarray,
            episode_starts: np.ndarray,
            info: InfoDict
    ) -> tuple[int, np.ndarray, np.ndarray]:
        with torch.no_grad():
            return super().perform_rollout(max_steps, obs, episode_starts, info)

    @override
    def optimize(
            self,
            last_obs: np.ndarray,
            last_episode_starts: np.ndarray,
            info: InfoDict
    ) -> None:
        last_values = self.policy.predict_values(
            torch.tensor(last_obs, device=self.torch_device, dtype=self.torch_dtype)
        )

        self.buffer.compute_returns_and_gae(
            last_values=last_values,
            last_episode_starts=last_episode_starts,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            normalize_rewards=self.normalize_rewards,
            normalize_advantages=self.normalize_advantages
        )

        optimization_infos: list[InfoDict] = []
        continue_training = True
        nr_updates = 0
        epochs_done = 0
        for i_epoch in range(self.ppo_max_epochs):
            epoch_infos: list[InfoDict] = []

            for batch_samples in self.buffer.get_samples(self.ppo_batch_size, self.shuffle_batches):
                batch_info: InfoDict = {}

                objectives = self.compute_ppo_objectives(
                    observations=batch_samples.observations,
                    advantages=batch_samples.advantages,
                    returns=batch_samples.returns,
                    old_actions=batch_samples.actions,
                    old_action_log_probs=batch_samples.old_log_probs,
                    old_value_estimates=batch_samples.old_values,
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

                grad_norm: float | None = None
                if self.grad_norm_clip_value:
                    grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_norm_clip_value).item()
                if self.logging_config.log_grad_norm:
                    if grad_norm is None:
                        grad_norm = calculate_grad_norm(self.policy)

                    batch_info['grad_norm'] = np.array([grad_norm])

                self.policy_optimizer.step()

                epoch_infos.append(batch_info)
                nr_updates += 1

            optimization_infos.append(concat_infos(epoch_infos))

            if continue_training:
                epochs_done += 1
            else:
                break

        for info_key, info_value in concat_infos(optimization_infos).items():
            info[info_key] = info_value

        info['nr_ppo_epochs'] = epochs_done
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
        new_action_selector, value_estimates = self.policy(observations)

        if self.logging_config.log_optimization_action_stds:
            info['action_stds'] = new_action_selector.distribution.stddev

        value_estimates = value_estimates.squeeze(dim=-1)

        actor_objective, entropy_objective = self.compute_ppo_actor_objectives(
            new_action_selector=new_action_selector,
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

        return [actor_objective, entropy_objective, critic_objective]

    def compute_ppo_actor_objectives(
            self,
            new_action_selector: ActionSelector,
            advantages: torch.Tensor,
            old_actions: torch.Tensor,
            old_action_log_probs: torch.Tensor,
            info: InfoDict,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        new_action_log_probs = new_action_selector.log_prob(old_actions)

        if self.ppo_kl_target is not None or self.logging_config.log_actor_kl_divergence:
            # https://github.com/DLR-RM/stable-baselines3/blob/285e01f64aa8ba4bd15aa339c45876d56ed0c3b4/stable_baselines3/ppo/ppo.py#L266
            with torch.no_grad():
                log_ratio = new_action_log_probs - old_action_log_probs
                approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().unsqueeze(0).numpy()

            if self.logging_config.log_actor_kl_divergence:
                info['actor_kl_divergence'] = approx_kl_div

            if self.ppo_kl_target is not None and approx_kl_div > 1.5 * self.ppo_kl_target:
                return None, None

        action_log_probs_ratios = torch.exp(new_action_log_probs - old_action_log_probs).sum(dim=-1)

        unclipped_actor_objective = advantages * action_log_probs_ratios
        clipped_actor_objective = advantages * torch.clamp(
            action_log_probs_ratios,
            1 - self.action_ratio_clip_range,
            1 + self.action_ratio_clip_range
        )
        raw_actor_objective = -torch.min(unclipped_actor_objective, clipped_actor_objective)

        actor_objective = weigh_and_reduce_objective(
            raw_objective=raw_actor_objective,
            weigh_and_reduce_function=self.weigh_and_reduce_actor_objective,
            info=info,
            objective_name='actor_objective',
            logging_config=self.logging_config.actor_objective,
        )

        if self.weigh_and_reduce_entropy_objective is not None:
            entropy = new_action_selector.entropy()
            if entropy is None:
                # Approximate entropy when no analytical form
                negative_entropy = new_action_log_probs
            else:
                negative_entropy = -entropy

            entropy_objective = weigh_and_reduce_objective(
                raw_objective=negative_entropy,
                weigh_and_reduce_function=self.weigh_and_reduce_entropy_objective,
                info=info,
                objective_name='entropy_objective',
                logging_config=self.logging_config.entropy_objective
            )
        else:
            entropy_objective = torch.zeros_like(actor_objective)

        return actor_objective, entropy_objective

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

        critic_objective = weigh_and_reduce_objective(
            raw_objective=raw_critic_objective,
            weigh_and_reduce_function=self.weigh_and_reduce_critic_objective,
            info=info,
            objective_name='critic_objective',
            logging_config=self.logging_config.critic_objective
        )

        return critic_objective
