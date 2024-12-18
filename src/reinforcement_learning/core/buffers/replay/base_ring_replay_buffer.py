import abc
from typing import NamedTuple

import numpy as np
import torch
from gymnasium import Env

from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.buffers.base_buffer import BaseBuffer
from src.reinforcement_learning.core.buffers.replay.base_replay_buffer import BaseReplayBuffer
from src.reinforcement_learning.core.type_aliases import ShapeDict, NpObs, TensorObs
from src.torch_device import TorchDevice


class ReplayBufferSamples(NamedTuple):
    observations: TensorObs
    actions: torch.Tensor
    next_observations: TensorObs
    dones: torch.Tensor
    rewards: torch.Tensor


class BaseRingReplayBuffer(BaseReplayBuffer, abc.ABC):

    def __init__(
            self,
            step_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...] | ShapeDict,
            action_shape: tuple[int, ...],
            reward_scale: float,
            consider_truncated_as_done: bool,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype,
            np_dtype: np.dtype,
    ):
        super().__init__(
            step_size=step_size,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_scale=reward_scale,
            consider_truncated_as_done=consider_truncated_as_done,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
        )

        self.actions = np.zeros((self.step_size, self.num_envs, *self.action_shape), dtype=self.np_dtype)

        self.rewards = np.zeros((self.step_size, self.num_envs), dtype=self.np_dtype)
        self.terminated = np.zeros((self.step_size, self.num_envs), dtype=bool)
        self.truncated = np.zeros((self.step_size, self.num_envs), dtype=bool)

    def add(
            self,
            observations: NpObs,
            next_observations: NpObs,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray,
    ) -> None:
        self._add_obs(observations=observations, next_observations=next_observations)

        self.actions[self.pos] = actions
        self.rewards[self.pos] = self.scale_rewards(rewards)
        self.terminated[self.pos] = terminated
        self.truncated[self.pos] = truncated

        self.pos += 1
        if self.pos == self.step_size:
            self.full = True
            self.pos = 0

    @abc.abstractmethod
    def _add_obs(
            self,
            observations: NpObs,
            next_observations: NpObs,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_batch_obs(self, step_indices: np.ndarray, env_indices: np.ndarray) -> tuple[TensorObs, TensorObs]:
        raise NotImplementedError

    def _get_batch(
            self,
            step_indices: np.ndarray,
            env_indices: np.ndarray,
    ) -> ReplayBufferSamples:
        tensor_obs, next_tensor_obs = self._get_batch_obs(step_indices, env_indices)

        if self.consider_truncated_as_done:
            dones = np.logical_or(self.terminated[step_indices, env_indices], self.truncated[step_indices, env_indices])
        else:
            dones = self.terminated[step_indices, env_indices]

        return ReplayBufferSamples(
            observations=tensor_obs,
            actions=self.to_torch(self.actions[step_indices, env_indices, :]),
            next_observations=next_tensor_obs,
            dones=self.to_torch(dones.reshape(-1, 1)),
            rewards=self.to_torch(self.rewards[step_indices, env_indices].reshape(-1, 1))
        )

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        env_indices = self.rng.choice(self.num_envs, batch_size)
        step_indices = self.rng.choice(self.step_count, batch_size)
        return self._get_batch(step_indices, env_indices)

    def compute_most_recent_episode_scores(
            self,
            n_episodes: int,
            compensate_for_reward_scaling: bool = True,
            consider_truncated_as_done: bool | None = None
    ):
        return self._compute_most_recent_episode_scores(
            rewards=self.rewards,
            terminated=self.terminated,
            truncated=self.truncated,
            n_episodes=n_episodes,
            compensate_for_reward_scaling=compensate_for_reward_scaling,
            consider_truncated_as_done=consider_truncated_as_done,
        )
