
from typing import Callable, Self

import gymnasium
import numpy as np
import torch
from overrides import override
from torch import optim

from src.reinforcement_learning.algorithms.policy_optimization_base import Policy, \
    PolicyProvider, LoggingConfig, PolicyOptimizationBase
from src.reinforcement_learning.algorithms.supervised_pre_training.supervised_pre_training import SupervisedPreTraining
from src.reinforcement_learning.core.buffers.basic_rollout_buffer import BasicRolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.infos import InfoDict
from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy
from src.torch_device import TorchDevice


class ZeroActionPreTraining(SupervisedPreTraining):
    def __init__(
            self,
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            policy_optimizer: optim.Optimizer | Callable[[ActorCriticPolicy], optim.Optimizer],
            buffer_size: int,
            callback: Callback[Self] = None,
            logging_config: LoggingConfig = None,
            torch_device: TorchDevice = 'cpu',
    ):
        super().__init__(
            compute_supervised_objective=self.compute_zero_action_objective,
            env=env,
            policy=policy,
            policy_optimizer=policy_optimizer,
            buffer_size=buffer_size,
            buffer_type=BasicRolloutBuffer,
            gamma=1.0,
            gae_lambda=1.0,
            sde_noise_sample_freq=None,
            reset_env_between_rollouts=False,
            callback=callback or Callback(),
            logging_config=logging_config,
            torch_device=torch_device,
        )

    @override
    def rollout_step(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        action_selector, _ = self.policy.process_obs(torch.tensor(obs, device=self.torch_device))
        actions = action_selector.get_actions()
        next_obs, rewards, terminated, truncated, info = self.env.step(actions.detach().cpu().numpy())

        if self.logging_config.log_rollout_action_stds:
            info['action_stds'] = action_selector.distribution.stddev

        self.buffer.add(
            observations=obs,
            rewards=rewards,
            episode_starts=np.logical_or(terminated, truncated),
            actions=actions,
            action_log_probs=action_selector.log_prob(actions),
        )

        return next_obs, rewards, terminated, truncated, info

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

    def compute_zero_action_objective(
            self,
            buffer: BasicRolloutBuffer,
            last_obs: np.ndarray,
            last_dones: np.ndarray,
            info: InfoDict
    ) -> torch.Tensor:
        obs = torch.tensor(buffer.observations, device=self.torch_device)
        action_selector, _ = self.policy.process_obs(obs)
        actions = action_selector.get_actions(deterministic=True)

        zero_action_objective = torch.mean(actions ** 2)

        info['zero_action_objective'] = zero_action_objective
        return zero_action_objective
