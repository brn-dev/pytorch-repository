from abc import ABC
from copy import deepcopy
from typing import TypeVar

import gymnasium
import numpy as np
import torch

from src.reinforcement_learning.algorithms.base.base_algorithm import BaseAlgorithm, Policy, LogConf, PolicyProvider
from src.reinforcement_learning.algorithms.base.on_policy_algorithm import RolloutBuf
from src.reinforcement_learning.core.action_noise import ActionNoise
from src.reinforcement_learning.core.buffers.replay.base_replay_buffer import BaseReplayBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.infos import InfoDict, stack_infos
from src.torch_device import TorchDevice
from src.void_list import VoidList

ReplayBuf = TypeVar('ReplayBuf', bound=BaseReplayBuffer)


class OffPolicyAlgorithm(BaseAlgorithm[Policy, ReplayBuf, LogConf], ABC):

    def __init__(
            self,
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            buffer: RolloutBuf,
            gamma: float,
            tau: float,
            rollout_steps: int,
            gradient_steps: int,
            action_noise: ActionNoise | None,
            warmup_steps: int,
            sde_noise_sample_freq: int | None,
            callback: Callback,
            logging_config: LogConf,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype
    ):
        assert action_noise is None or isinstance(self.env.action_space, gymnasium.spaces.Box), \
                'Can only use action noise with continuous actions!'

        super().__init__(
            env=env,
            policy=policy,
            buffer=buffer,
            gamma=gamma,
            sde_noise_sample_freq=sde_noise_sample_freq,
            callback=callback,
            logging_config=logging_config,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
        )

        self.tau = tau

        self.rollout_steps = rollout_steps
        self.gradient_steps = gradient_steps

        self.action_noise = action_noise
        self.warmup_steps = warmup_steps

        self.steps_performed = 0

    def _on_step(self):
        pass

    def sample_actions(self, obs: np.ndarray, info: InfoDict):
        if self.steps_performed < self.warmup_steps:
            actions = np.array([self.action_space.sample() for _ in range(self.num_envs)])
        else:
            action_selector = self.policy.act(
                torch.tensor(obs, device=self.torch_device, dtype=self.torch_dtype)
            )

            if self.logging_config.log_rollout_action_stds:
                info['action_stds'] = action_selector.distribution.stddev

            actions = action_selector.get_actions().cpu().numpy()

        if self.action_noise is not None:
            actions = self.clip_actions(actions + self.action_noise())

        return actions

    def rollout_step(
            self,
            obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        info: InfoDict = {}

        actions = self.sample_actions(obs, info)

        new_obs, rewards, terminated, truncated, step_info = self.env.step(actions)
        dones = np.logical_or(terminated, truncated)
        info.update(step_info)

        next_obs = deepcopy(new_obs)

        final_observations = step_info.get('final_observation')
        for i, done in enumerate(dones):
            if done and (final_obs := final_observations[i]) is not None:
                if isinstance(next_obs, dict):
                    for key in next_obs.keys():
                        next_obs[key][i] = final_obs[key]
                else:
                    next_obs[i] = final_obs

        print(f'{obs = }')
        print(f'{next_obs = }')
        print()

        self.buffer.add(
            observations=obs,
            next_observations=next_obs,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

        self._on_step()

        self.steps_performed += 1
        return new_obs, np.logical_or(terminated, truncated), info

    def perform_rollout(
            self,
            max_steps: int,
            obs: np.ndarray,
            episode_starts: np.ndarray,
            info: InfoDict
    ) -> tuple[int, np.ndarray, np.ndarray]:
        self.policy.reset_sde_noise(self.num_envs)

        infos: list[InfoDict] = [] if self.logging_config.log_rollout_infos else VoidList()
        step = 0

        for step in range(min(self.rollout_steps, max_steps)):
            if self.sde_noise_sample_freq is not None and step % self.sde_noise_sample_freq == 0:
                self.policy.reset_sde_noise(self.num_envs)

            obs, episode_starts, step_info = self.rollout_step(obs)
            infos.append(step_info)

            if np.any(episode_starts) and self.action_noise is not None:
                self.action_noise.reset(np.where(episode_starts)[0])

        if self.logging_config.log_rollout_infos:
            info['rollout'] = stack_infos(infos)

        if self.logging_config.log_last_obs:
            info['last_obs'] = obs
            info['last_episode_starts'] = episode_starts

        return step + 1, obs, episode_starts
