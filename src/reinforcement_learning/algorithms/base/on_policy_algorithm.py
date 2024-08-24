import abc
from typing import Callable, TypeVar, Generic

import gymnasium
import numpy as np
import torch
from torch import optim

from src.reinforcement_learning.algorithms.base.logging_config import LoggingConfig
from src.reinforcement_learning.core.action_selectors.continuous_action_selector import ContinuousActionSelector
from src.reinforcement_learning.core.buffers.rollout.rollout_buffer import RolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.infos import InfoDict, stack_infos
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.gym.singleton_vector_env import as_vec_env
from src.torch_device import TorchDevice, optimizer_to_device

LogConf = TypeVar('LogConf', bound=LoggingConfig)

_RolloutBuffer = TypeVar('_RolloutBuffer', bound=RolloutBuffer)

Policy = TypeVar('Policy', bound=BasePolicy)
PolicyProvider = Callable[[], Policy]


class OnPolicyAlgorithm(Generic[Policy, _RolloutBuffer, LogConf], abc.ABC):

    def __init__(
            self,
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            policy_optimizer: optim.Optimizer | Callable[[BasePolicy], optim.Optimizer],
            buffer: _RolloutBuffer,
            gamma: float,
            gae_lambda: float,
            sde_noise_sample_freq: int | None,
            reset_env_between_rollouts: bool,
            callback: Callback,
            logging_config: LogConf,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype
    ):
        self.env, self.num_envs = as_vec_env(env)

        self.policy: Policy = (policy if isinstance(policy, BasePolicy) else policy()).to(torch_device)
        self.buffer = buffer

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        if not policy.uses_sde and sde_noise_sample_freq is not None:
            print(f'================================= Warning ================================= \n'
                  f' SDE noise sample freq is set to {sde_noise_sample_freq} despite not using SDE \n'
                  f'=========================================================================== \n\n\n')
        if policy.uses_sde and sde_noise_sample_freq is None:
            raise ValueError(f'SDE noise sample freq is set to None despite using SDE')

        self.sde_noise_sample_freq = sde_noise_sample_freq

        self.logging_config = logging_config
        if (self.logging_config.log_rollout_action_stds
                and not isinstance(policy.action_selector, ContinuousActionSelector)):
            raise ValueError('Cannot log action distribution stds with non continuous action selector')

        if isinstance(policy_optimizer, optim.Optimizer):
            self.policy_optimizer = optimizer_to_device(policy_optimizer, torch_device)
        else:
            self.policy_optimizer = optimizer_to_device(policy_optimizer(policy), torch_device)

        self.reset_env_between_rollouts = reset_env_between_rollouts
        self.callback = callback

        self.torch_device = torch_device
        self.torch_dtype = torch_dtype

    @abc.abstractmethod
    def optimize(
            self,
            last_obs: np.ndarray,
            last_episode_starts: np.ndarray,
            info: InfoDict
    ) -> None:
        raise NotImplemented

    def rollout_step(
            self,
            obs: np.ndarray,
            episode_starts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        action_selector, value_estimates = self.policy(
            torch.tensor(obs, device=self.torch_device, dtype=self.torch_dtype)
        )
        actions = action_selector.get_actions()
        next_obs, rewards, terminated, truncated, info = self.env.step(actions.detach().cpu().numpy())

        if self.logging_config.log_rollout_action_stds:
            info['action_stds'] = action_selector.distribution.stddev

        self.buffer.add(
            observations=obs,
            rewards=rewards,
            episode_starts=episode_starts,
            actions=actions,
            action_log_probs=action_selector.log_prob(actions),
            value_estimates=value_estimates
        )

        return next_obs, rewards, np.logical_or(terminated, truncated), info

    def perform_rollout(
            self,
            max_steps: int,
            obs: np.ndarray,
            episode_starts: np.ndarray,
            info: InfoDict
    ) -> tuple[int, np.ndarray, np.ndarray]:
        step = 0

        self.policy.reset_sde_noise(self.num_envs)

        infos: list[InfoDict] = []
        for step in range(min(self.buffer.buffer_size, max_steps)):
            if self.sde_noise_sample_freq is not None and step % self.sde_noise_sample_freq == 0:
                self.policy.reset_sde_noise(self.num_envs)

            obs, rewards, episode_starts, step_info = self.rollout_step(obs, episode_starts)
            infos.append(step_info)

        if self.logging_config.log_rollout_infos:
            info['rollout'] = stack_infos(infos)

        if self.logging_config.log_last_obs:
            info['last_obs'] = obs
            info['last_episode_starts'] = episode_starts

        return step + 1, obs, episode_starts

    def train(self, num_steps: int):
        self.policy.train()

        obs: np.ndarray = np.empty(())
        episode_starts: np.ndarray = np.empty(())

        step = 0
        while step < num_steps:
            info: InfoDict = {}

            if step == 0 or self.reset_env_between_rollouts:
                obs, reset_info = self.env.reset()
                episode_starts = np.ones(self.num_envs, dtype=bool)

                if self.logging_config.log_reset_info:
                    info['reset'] = reset_info

            steps_performed, obs, episode_starts = self.perform_rollout(
                max_steps=num_steps - step,
                obs=obs,
                episode_starts=episode_starts,
                info=info
            )
            step += steps_performed

            self.callback.on_rollout_done(self, step, info)

            self.optimize(obs, episode_starts, info)

            self.callback.on_optimization_done(self, step, info)

            self.buffer.reset()
