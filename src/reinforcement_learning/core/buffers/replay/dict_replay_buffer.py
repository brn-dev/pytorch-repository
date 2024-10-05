from typing import NamedTuple

import numpy as np
import torch

from src.reinforcement_learning.core.buffers.replay.base_replay_buffer import BaseReplayBuffer, ReplayBufferSamples
from src.reinforcement_learning.core.type_aliases import TensorDict, ShapeDict, NpArrayDict
from src.torch_device import TorchDevice


class DictReplayBuffer(BaseReplayBuffer):

    def __init__(
            self,
            buffer_size: int,
            num_envs: int,
            obs_shape: ShapeDict,
            action_shape: tuple[int, ...],
            reward_scale: float,
            torch_device: TorchDevice = 'cpu',
            torch_dtype: torch.dtype = torch.float32,
            np_dtype: np.dtype = np.float32,
    ):
        super().__init__(
            buffer_size=buffer_size,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_scale=reward_scale,
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

    def _add_obs(
        self,
        observations: NpArrayDict,
        next_observations: NpArrayDict,
    ) -> None:
        for key in self.obs_keys:
            self.observations[key][self.pos] = observations[key]
            self.next_observations[key][self.pos] = next_observations[key]

    def _get_batch(self, step_indices: np.ndarray, env_indices: np.ndarray) -> ReplayBufferSamples:
        obs = {key: self.to_torch(_obs[step_indices, env_indices, :]) for key, _obs in self.observations.items()}
        next_obs = {
            key: self.to_torch(_next_obs[step_indices, env_indices, :])
            for key, _next_obs in self.next_observations.items()
        }

        return ReplayBufferSamples(
            observations=obs,
            actions=self.to_torch(self.actions[step_indices, env_indices, :]),
            next_observations=next_obs,
            dones=self.to_torch(self.dones[step_indices, env_indices].reshape(-1, 1)),
            rewards=self.to_torch(self.rewards[step_indices, env_indices].reshape(-1, 1))
        )
