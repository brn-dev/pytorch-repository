from typing import NamedTuple

import numpy as np
import torch

from src.reinforcement_learning.core.buffers.replay.base_replay_buffer import BaseReplayBuffer
from src.torch_device import TorchDevice
from src.type_aliases import TensorDict, ShapeDict, NpArrayDict


class DictReplayBufferSamples(NamedTuple):
    observations: TensorDict
    actions: torch.Tensor
    next_observations: TensorDict
    dones: torch.Tensor
    rewards: torch.Tensor


class DictReplayBuffer(BaseReplayBuffer[DictReplayBufferSamples]):

    def __init__(
            self,
            buffer_size: int,
            num_envs: int,
            obs_shape: ShapeDict,
            action_shape: tuple[int, ...],
            torch_device: TorchDevice = 'cpu',
            torch_dtype: torch.dtype = torch.float32,
            np_dtype: np.dtype = np.float32,
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
        assert isinstance(self.obs_shape, dict), "DictReplayBuffer must be used with dict obs shape only"

        self.observations = {
            key: np.zeros((self.buffer_size, self.num_envs, *_obs_shape), dtype=np_dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.next_observations = {
            key: np.zeros((self.buffer_size, self.num_envs, *_obs_shape), dtype=np_dtype)
            for key, _obs_shape in self.obs_shape.items()
        }
        self.obs_keys = set(self.obs_shape.keys())

        self.actions = np.zeros((self.buffer_size, self.num_envs, *self.action_shape), dtype=self.np_dtype)

        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=self.np_dtype)
        self.dones = np.zeros((self.buffer_size, self.num_envs), dtype=bool)

    def add(  # type: ignore[override]
        self,
        obs: NpArrayDict,
        next_obs: NpArrayDict,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        for key in self.obs_keys:
            self.observations[key][self.pos] = obs[key]
            self.next_observations[key][self.pos] = next_obs[key]

        self.actions[self.pos] = actions
        self.rewards[self.pos] = rewards
        self.dones[self.pos] = dones

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int, with_replacement: bool = False):
        env_indices = np.random.randint(0, high=self.num_envs, size=batch_size)
        step_indices = np.random.choice(self.size, batch_size, replace=with_replacement)
        return self.get_batch(step_indices, env_indices)

    def get_batch(self, step_indices: np.ndarray, env_indices: np.ndarray) -> DictReplayBufferSamples:
        obs = {key: self.to_torch(_obs[step_indices, env_indices, :]) for key, _obs in self.observations.items()}
        next_obs = {
            key: self.to_torch(_next_obs[step_indices, env_indices, :])
            for key, _next_obs in self.next_observations.items()
        }

        return DictReplayBufferSamples(
            observations=obs,
            actions=self.to_torch(self.actions[step_indices, env_indices, :]),
            next_observations=next_obs,
            dones=self.to_torch(self.dones[step_indices, env_indices].reshape(-1, 1)),
            rewards=self.to_torch(self.rewards[step_indices, env_indices].reshape(-1, 1))
        )
