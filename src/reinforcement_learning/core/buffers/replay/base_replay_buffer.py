import abc
from typing import NamedTuple

import numpy as np
import torch

from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.buffers.base_buffer import BaseBuffer
from src.reinforcement_learning.core.type_aliases import ShapeDict, NpObs, TensorObs
from src.torch_device import TorchDevice


class ReplayBufferSamples(NamedTuple):
    observations: TensorObs
    actions: torch.Tensor
    next_observations: TensorObs
    dones: torch.Tensor
    rewards: torch.Tensor


class BaseReplayBuffer(BaseBuffer[ReplayBufferSamples], abc.ABC):

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
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
        )

        self.actions = np.zeros((self.step_size, self.num_envs, *self.action_shape), dtype=self.np_dtype)

        self.rewards = np.zeros((self.step_size, self.num_envs), dtype=self.np_dtype)
        self.dones = np.zeros((self.step_size, self.num_envs), dtype=bool)
        self.truncated = np.zeros((self.step_size, self.num_envs), dtype=bool)

        self.consider_truncated_as_done = consider_truncated_as_done

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'consider_truncated_as_done': self.consider_truncated_as_done,
        })

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
        self.dones[self.pos] = np.logical_or(terminated, truncated)
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
            dones = self.dones[step_indices, env_indices]
        else:
            dones = np.logical_and(
                self.dones[step_indices, env_indices],
                np.logical_not(self.truncated[step_indices, env_indices])
            )

        return ReplayBufferSamples(
            observations=tensor_obs,
            actions=self.to_torch(self.actions[step_indices, env_indices, :]),
            next_observations=next_tensor_obs,
            dones=self.to_torch(dones.reshape(-1, 1)),
            rewards=self.to_torch(self.rewards[step_indices, env_indices].reshape(-1, 1))
        )

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.num_envs, size=batch_size)
        step_indices = np.random.choice(self.size, batch_size)
        return self._get_batch(step_indices, env_indices)

    def tail_indices(self, tail_length: int):
        if tail_length > self.size:
            tail_length = self.size

        if not self.full or self.pos >= tail_length:
            return np.arange(self.pos - tail_length, self.pos)

        return np.concatenate((
            np.arange(self.size - tail_length + self.pos, self.size),
            np.arange(self.pos)
        ))

    def compute_most_recent_episode_scores(
            self,
            n_episodes: int,
            compensate_for_reward_scaling: bool = True,
    ):
        whole_episode = np.zeros((self.num_envs,), dtype=bool)

        running_sum = np.zeros((self.num_envs,), dtype=float)
        episode_scores: list[float] = []

        for step_index in reversed(self.tail_indices(self.size)):
            step_dones = self.dones[step_index]
            episode_scores.extend(running_sum[np.logical_and(step_dones, whole_episode)])

            if len(episode_scores) >= n_episodes:
                break
                
            whole_episode[step_dones] = True
            running_sum[step_dones] = 0.0

            step_rewards = self.rewards[step_index]
            if compensate_for_reward_scaling:
                step_rewards = self.unscale_rewards(step_rewards)

            running_sum += step_rewards

        return np.array(episode_scores)
