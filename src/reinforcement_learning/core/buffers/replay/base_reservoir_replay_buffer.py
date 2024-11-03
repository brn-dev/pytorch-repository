import abc

import numpy as np
import torch
from gymnasium import Env

from src.reinforcement_learning.core.buffers.replay.base_replay_buffer import ReplayBufferSamples, BaseReplayBuffer
from src.reinforcement_learning.core.type_aliases import ShapeDict, NpObs, TensorObs
from src.reinforcement_learning.gym.env_analysis import get_num_envs, get_obs_shape, get_action_shape
from src.torch_device import TorchDevice

"""

        https://en.wikipedia.org/wiki/Reservoir_sampling

"""
class BaseReservoirReplayBuffer(BaseReplayBuffer, abc.ABC):

    def __init__(
            self,
            total_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...] | ShapeDict,
            action_shape: tuple[int, ...],
            reward_scale: float,
            consider_truncated_as_done: bool,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype,
            np_dtype: np.dtype,
    ):
        assert total_size % num_envs == 0
        super().__init__(
            step_size=total_size // num_envs,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_scale=reward_scale,
            consider_truncated_as_done=consider_truncated_as_done,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
        )
        self.total_size = total_size

        self.actions = np.zeros((self.total_size, *self.action_shape), dtype=self.np_dtype)

        self.rewards = np.zeros(self.total_size, dtype=self.np_dtype)
        self.terminated = np.zeros(self.total_size, dtype=bool)
        self.truncated = np.zeros(self.total_size, dtype=bool)

    @property
    def step_count(self):
        return self.total_size if self.full else self.pos

    @property
    def count(self):
        return self.step_count

    def add(
            self,
            observations: NpObs,
            next_observations: NpObs,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray,
    ) -> None:
        if not self.full:
            self._add_at(
                indices=np.arange(self.pos, self.pos + self.num_envs),
                observations=observations,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
            )
        else:
            indices = self.rng.integers(0, self.pos + 1 + np.arange(self.num_envs))
            accepted = indices < self.total_size
            if np.any(accepted):
                self._add_at(
                    indices=indices[accepted],
                    observations=observations[accepted],
                    next_observations=next_observations[accepted],
                    actions=actions[accepted],
                    rewards=rewards[accepted],
                    terminated=terminated[accepted],
                    truncated=truncated[accepted],
                )

        self.pos += self.num_envs
        if self.pos == self.total_size:
            self.full = True

    def _add_at(
            self,
            indices: np.ndarray,
            observations: NpObs,
            next_observations: NpObs,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray,
    ):
        self._add_obs_at(indices, observations=observations, next_observations=next_observations)

        self.actions[indices] = actions
        self.rewards[indices] = self.scale_rewards(rewards)
        self.terminated[indices] = terminated
        self.truncated[indices] = truncated

    @abc.abstractmethod
    def _add_obs_at(
            self,
            indices: np.ndarray,
            observations: NpObs,
            next_observations: NpObs,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _get_batch_obs(self, batch_indices: np.ndarray) -> tuple[TensorObs, TensorObs]:
        raise NotImplementedError

    def _get_batch(self, batch_indices: np.ndarray) -> ReplayBufferSamples:
        tensor_obs, next_tensor_obs = self._get_batch_obs(batch_indices)

        if self.consider_truncated_as_done:
            dones = np.logical_or(self.terminated[batch_indices], self.truncated[batch_indices])
        else:
            dones = self.terminated[batch_indices]

        return ReplayBufferSamples(
            observations=tensor_obs,
            actions=self.to_torch(self.actions[batch_indices, :]),
            next_observations=next_tensor_obs,
            dones=self.to_torch(dones.reshape(-1, 1)),
            rewards=self.to_torch(self.rewards[batch_indices].reshape(-1, 1))
        )

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        batch_indices = self.rng.choice(self.step_count, batch_size)
        return self._get_batch(batch_indices)

    def tail_indices(self, tail_length: int):
        if self.pos > self.step_size:
            raise RuntimeError('Can not get tail indices from an over-full reservoir buffer')
        return super().tail_indices(tail_length)

    def compute_most_recent_episode_scores(
            self,
            n_episodes: int,
            compensate_for_reward_scaling: bool = True,
            consider_truncated_as_done: bool | None = None
    ):
        return self._compute_most_recent_episode_scores(
            rewards=self.rewards.reshape(-1, self.num_envs),
            terminated=self.terminated.reshape(-1, self.num_envs),
            truncated=self.truncated.reshape(-1, self.num_envs),
            n_episodes=n_episodes,
            compensate_for_reward_scaling=compensate_for_reward_scaling,
            consider_truncated_as_done=consider_truncated_as_done,
        )

    # noinspection PyMethodOverriding
    @classmethod
    def for_env(
            cls,
            env: Env,
            buffer_size: int,
            reward_scale: float,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype = torch.float32,
            np_dtype: np.dtype = np.float32,
            **buffer_kwargs
    ):
        num_envs = get_num_envs(env)

        obs_shape = get_obs_shape(env)
        action_shape = get_action_shape(env)

        return cls(
            total_size=buffer_size,
            num_envs=num_envs,
            reward_scale=reward_scale,
            obs_shape=obs_shape,
            action_shape=action_shape,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
            **buffer_kwargs
        )
