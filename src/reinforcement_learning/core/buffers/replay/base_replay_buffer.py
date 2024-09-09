import abc

import numpy as np
import torch

from src.reinforcement_learning.core.buffers.base_buffer import BaseBuffer, BufferSamples
from src.reinforcement_learning.core.type_aliases import ShapeDict, NpObs
from src.torch_device import TorchDevice


class BaseReplayBuffer(BaseBuffer[BufferSamples], abc.ABC):

    def __init__(
            self,
            buffer_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...] | ShapeDict,
            action_shape: tuple[int, ...],
            torch_device: TorchDevice,
            torch_dtype: torch.dtype,
            np_dtype: np.dtype,
    ):
        super().__init__(
            buffer_size=buffer_size,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
        )

        self.actions = np.zeros((self.buffer_size, self.num_envs, *self.action_shape), dtype=self.np_dtype)

        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=self.np_dtype)
        self.dones = np.zeros((self.buffer_size, self.num_envs), dtype=bool)

        # TODO: maybe introduce truncation logic
        # https://github.com/DLR-RM/stable-baselines3/blob/9a3b28bb9f24a1646479500fb23be55ba652a30d/stable_baselines3/common/buffers.py#L321

    @abc.abstractmethod
    def add(
            self,
            observations: NpObs,
            next_observations: NpObs,
            actions: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray,
    ) -> None:
        raise NotImplementedError

    def _add(
            self,
            actions: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray
    ):
        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    @abc.abstractmethod
    def sample(self, batch_size: int) -> BufferSamples:
        raise NotImplementedError
