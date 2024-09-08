from typing import Generator, NamedTuple

import numpy as np
import torch

from src.reinforcement_learning.core.buffers.rollout.base_rollout_buffer import BaseRolloutBuffer
from src.torch_device import TorchDevice


class RolloutBufferSamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    old_values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor


class RolloutBuffer(BaseRolloutBuffer[RolloutBufferSamples]):

    observations: np.ndarray

    flat_observations: np.ndarray
    flat_actions: np.ndarray
    flat_action_log_probs: np.ndarray
    flat_value_estimates: np.ndarray
    flat_returns: np.ndarray
    flat_advantages: np.ndarray

    def __init__(
            self,
            buffer_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...],
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

        self.observations = np.zeros((self.buffer_size, self.num_envs, *self.obs_shape), dtype=self.np_dtype)
        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=self.np_dtype)
        self.episode_starts = np.zeros((self.buffer_size, self.num_envs), dtype=bool)

        self.actions = np.zeros((self.buffer_size, self.num_envs, *self.action_shape), dtype=self.np_dtype)
        self.action_log_probs = np.zeros((self.buffer_size, self.num_envs), dtype=self.np_dtype)

        self.value_estimates = np.zeros((self.buffer_size, self.num_envs), dtype=self.np_dtype)

        self.reset()

    def reset(self):
        super().reset()

        self.returns = None
        self.advantages = None

    def add(
            self,
            observations: np.ndarray,
            rewards: np.ndarray,
            episode_starts: np.ndarray,
            actions: torch.Tensor,
            action_log_probs: torch.Tensor,
            value_estimates: torch.Tensor,
    ):
        assert not self.full

        self.observations[self.pos] = observations
        self.rewards[self.pos] = rewards
        self.episode_starts[self.pos] = episode_starts

        self.actions[self.pos] = actions.cpu().numpy()
        self.action_log_probs[self.pos] = action_log_probs.cpu().numpy()

        self.value_estimates[self.pos] = value_estimates.squeeze(-1).cpu().numpy()

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get_samples(
            self,
            batch_size: int | None = None
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.advantages is not None and self.returns is not None, \
            'Call compute_gae_and_returns before calling get_samples'

        num_samples = self.pos * self.num_envs

        indices = np.random.permutation(num_samples)

        if batch_size is None:
            batch_size = num_samples

        self.create_flattened_array_views()

        for start_index in range(0, num_samples, batch_size):
            yield self.get_batch(indices[start_index:start_index + batch_size])

    def create_flattened_array_views(self):
        self.flat_observations = self.flatten(self.observations)
        self.flat_actions = self.flatten(self.actions)
        self.flat_action_log_probs = self.flatten(self.action_log_probs)
        self.flat_value_estimates = self.flatten(self.value_estimates)
        self.flat_returns = self.flatten(self.returns)
        self.flat_advantages = self.flatten(self.advantages)

    def get_batch(self, batch_indices: np.ndarray):
        data = (
            self.flat_observations[batch_indices],
            self.flat_actions[batch_indices],
            self.flat_action_log_probs[batch_indices],
            self.flat_value_estimates[batch_indices],
            self.flat_returns[batch_indices],
            self.flat_advantages[batch_indices],
        )
        return RolloutBufferSamples(*self.all_to_torch(data))
