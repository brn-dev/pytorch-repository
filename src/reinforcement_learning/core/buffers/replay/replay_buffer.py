import numpy as np
import torch

from src.reinforcement_learning.core.buffers.replay.base_replay_buffer import BaseReplayBuffer, ReplayBufferSamples
from src.torch_device import TorchDevice


class ReplayBuffer(BaseReplayBuffer):

    def __init__(
            self,
            buffer_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...],
            action_shape: tuple[int, ...],
            optimize_memory_usage: bool = False,
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

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.num_envs, *self.obs_shape), dtype=self.np_dtype)
        if not optimize_memory_usage:
            self.next_observations = np.zeros((self.buffer_size, self.num_envs, *obs_shape), dtype=self.np_dtype)

    def add(
        self,
        observations: np.ndarray,
        next_observations: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        self.observations[self.pos] = observations

        # TODO: using this simple logic when optimize_memory_usage=True stores and samples the transition
        # TODO: last obs before terminal -> first obs of new episode
        # TODO: which preferably should not happen
        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = next_observations
        else:
            self.next_observations[self.pos] = next_observations

        self._add(
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.num_envs, size=batch_size)

        if not self.optimize_memory_usage:
            step_indices = np.random.choice(self.size, batch_size)
            return self.get_batch(step_indices, env_indices)

        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            step_indices = (
               np.random.choice(self.buffer_size - 1, batch_size) + self.pos + 1
            ) % self.buffer_size
        else:
            step_indices = np.random.choice(self.pos, batch_size)

        return self.get_batch(step_indices, env_indices)

    def get_batch(self, step_indices: np.ndarray, env_indices: np.ndarray) -> ReplayBufferSamples:
        if self.optimize_memory_usage:
            next_obs = self.observations[(step_indices + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[step_indices, env_indices, :]

        data = (
            self.observations[step_indices, env_indices, :],
            self.actions[step_indices, env_indices, :],
            next_obs,
            self.dones[step_indices, env_indices].reshape(-1, 1),
            self.rewards[step_indices, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*self.all_to_torch(data))
