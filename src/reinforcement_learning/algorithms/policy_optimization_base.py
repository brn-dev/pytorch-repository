import abc
from dataclasses import dataclass
from typing import Callable, TypeVar

import gymnasium
import numpy as np
import torch
from torch import optim

from src.reinforcement_learning.core.action_selectors.continuous_action_selector import ContinuousActionSelector
from src.reinforcement_learning.core.buffers.basic_rollout_buffer import BasicRolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.generalized_advantage_estimate import compute_episode_returns
from src.reinforcement_learning.core.infos import InfoDict, stack_infos
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.gym.singleton_vector_env import as_vec_env
from src.reinforcement_learning.gym.wrappers.reward_wrapper import RewardWrapper
from src.torch_device import TorchDevice, optimizer_to_device


@dataclass
class RolloutLoggingConfig:
    log_rollout_infos: bool = False
    log_rollout_action_stds: bool = False
    log_last_obs: bool = False

    def __post_init__(self):
        assert not self.log_rollout_action_stds or self.log_rollout_infos, \
            'log_rollout_infos has to be enabled for log_rollout_stds'


@dataclass
class LoggingConfig(RolloutLoggingConfig):
    log_reset_info: bool = False


LogConf = TypeVar('LogConf', bound=LoggingConfig)
RolloutLogConf = TypeVar('RolloutLogConf', bound=RolloutLoggingConfig)

Buffer = TypeVar('Buffer', bound=BasicRolloutBuffer)

Policy = TypeVar('Policy', bound=BasePolicy)
PolicyProvider = Callable[[], Policy]


class PolicyRollout:

    def __init__(
            self,
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            buffer: Buffer,
            gamma: float,
            gae_lambda: float,
            sde_noise_sample_freq: int | None,
            logging_config: RolloutLogConf,
            torch_device: TorchDevice,
    ):
        self.env, self.num_envs = as_vec_env(env)

        self.policy = (policy if isinstance(policy, BasePolicy) else policy()).to(torch_device)
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

        self.torch_device = torch_device

    def rollout_step(
            self,
            obs: np.ndarray,
            episode_starts: np.ndarray,
            buffer: Buffer,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        action_selector, extra_predictions = self.policy.process_obs(torch.tensor(obs, device=self.torch_device))
        actions = action_selector.get_actions()
        next_obs, rewards, terminated, truncated, info = self.env.step(actions.detach().cpu().numpy())

        if self.logging_config.log_rollout_action_stds:
            info['action_stds'] = action_selector.distribution.stddev

        buffer.add(
            observations=obs,
            rewards=rewards,
            episode_starts=episode_starts,
            actions=actions,
            action_log_probs=action_selector.log_prob(actions),
            **extra_predictions
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

            obs, rewards, episode_starts, step_info = self.rollout_step(obs, episode_starts, self.buffer)
            infos.append(step_info)

        if self.logging_config.log_rollout_infos:
            info['rollout'] = stack_infos(infos)

        if self.logging_config.log_last_obs:
            info['last_obs'] = obs
            info['last_episode_starts'] = episode_starts

        return step + 1, obs, episode_starts

    def evaluate(
            self,
            num_steps: int,
            gamma: float | None = None,
            gae_lambda: float | None = None,
            remove_unfinished_episodes: bool = True,
    ) -> tuple[BasicRolloutBuffer, InfoDict, np.ndarray, np.ndarray]:
        infos: list[InfoDict] = []
        buffer = BasicRolloutBuffer(
            buffer_size=num_steps,
            num_envs=self.num_envs,
            obs_shape=self.env.observation_space.shape
        )

        obs, _ = self.env.reset()
        episode_starts = np.ones(self.num_envs, dtype=bool)

        for step in range(num_steps):
            obs, _, episode_starts, info = self.rollout_step(obs, episode_starts, buffer)
            infos.append(info)

        combined_info = stack_infos(infos)

        rewards = buffer.rewards
        if RewardWrapper.RAW_REWARDS_KEY in combined_info:
            rewards = combined_info[RewardWrapper.RAW_REWARDS_KEY]

        episode_returns = compute_episode_returns(
            rewards=rewards,
            episode_starts=buffer.episode_starts,
            last_episode_starts=episode_starts,
            gamma=gamma if gamma is not None else self.gamma,
            gae_lambda=gae_lambda if gae_lambda is not None else self.gae_lambda,
            normalize_rewards=None,
            remove_unfinished_episodes=remove_unfinished_episodes,
        )

        return buffer, combined_info, rewards, episode_returns


class PolicyOptimizationBase(PolicyRollout, abc.ABC):

    def __init__(
            self,
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            policy_optimizer: optim.Optimizer | Callable[[BasePolicy], optim.Optimizer],
            buffer: Buffer,
            gamma: float,
            gae_lambda: float,
            sde_noise_sample_freq: int | None,
            reset_env_between_rollouts: bool,
            callback: Callback,
            logging_config: LogConf,
            torch_device: TorchDevice,
    ):
        super().__init__(
            env=env,
            policy=policy,
            buffer=buffer,
            gamma=gamma,
            gae_lambda=gae_lambda,
            sde_noise_sample_freq=sde_noise_sample_freq,
            logging_config=logging_config,
            torch_device=torch_device,
        )
        if isinstance(policy_optimizer, optim.Optimizer):
            self.policy_optimizer = optimizer_to_device(policy_optimizer, torch_device)
        else:
            self.policy_optimizer = optimizer_to_device(policy_optimizer(policy), torch_device)

        self.reset_env_between_rollouts = reset_env_between_rollouts
        self.callback = callback

    @abc.abstractmethod
    def optimize(
            self,
            last_obs: np.ndarray,
            last_episode_starts: np.ndarray,
            info: InfoDict
    ) -> None:
        raise NotImplemented

    def train(self, num_steps: int):
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

