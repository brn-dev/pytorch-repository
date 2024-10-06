import abc
from typing import NamedTuple, Callable

import numpy as np
import torch

from src.reinforcement_learning.core.buffers.base_buffer import BaseBuffer
from src.reinforcement_learning.core.type_aliases import ShapeDict, NpObs, TensorObs
from src.torch_device import TorchDevice


class ReservoirBufferSamples(NamedTuple):
    observations: TensorObs
    actions: torch.Tensor
    next_observations: TensorObs
    dones: torch.Tensor
    rewards: torch.Tensor

"""

        https://en.wikipedia.org/wiki/Reservoir_sampling

"""
class BaseReservoirBuffer(BaseBuffer[ReservoirBufferSamples], abc.ABC):

    def __init__(
            self,
            total_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...] | ShapeDict,
            action_shape: tuple[int, ...],
            reward_scale: float,
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
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
        )
        self.total_size = total_size

        self.actions = np.zeros((self.total_size, *self.action_shape), dtype=self.np_dtype)

        self.rewards = np.zeros(self.total_size, dtype=self.np_dtype)
        self.dones = np.zeros(self.total_size, dtype=bool)

        self.rng = np.random.default_rng(None)

    @property
    def size(self):
        return self.total_size if self.full else self.pos

    def add(
            self,
            observations: NpObs,
            next_observations: NpObs,
            actions: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray,
    ) -> None:
        if not self.full:
            self._add_at(
                indices=np.arange(self.pos, self.pos + self.num_envs),
                observations=observations,
                next_observations=next_observations,
                actions=actions,
                rewards=rewards,
                dones=dones,
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
                    dones=dones[accepted],
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
            dones: np.ndarray,
    ):
        self._add_obs_at(indices, observations=observations, next_observations=next_observations)

        self.actions[indices] = actions
        self.rewards[indices] = self.scale_rewards(rewards)
        self.dones[indices] = dones

    @abc.abstractmethod
    def _add_obs_at(
            self,
            indices: np.ndarray,
            observations: NpObs,
            next_observations: NpObs,
    ) -> None:
        raise NotImplementedError

    def sample(self, batch_size: int) -> ReservoirBufferSamples:
        batch_indices = np.random.choice(self.size, batch_size)
        return self._get_batch(batch_indices)

    @abc.abstractmethod
    def _get_batch(self, batch_indices: np.ndarray) -> ReservoirBufferSamples:
        raise NotImplementedError

