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

    samples_ready: bool

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
        self.reset()

    def reset(self):
        super().reset()
        self.observations = np.zeros((self.buffer_size, self.num_envs, *self.obs_shape), dtype=self.np_dtype)
        self.rewards = np.zeros((self.buffer_size, self.num_envs), dtype=self.np_dtype)
        self.episode_starts = np.zeros((self.buffer_size, self.num_envs), dtype=bool)

        self.actions = np.zeros((self.buffer_size, self.num_envs, *self.action_shape), dtype=self.np_dtype)
        self.action_log_probs = np.zeros((self.buffer_size, self.num_envs), dtype=self.np_dtype)

        self.value_estimates = np.zeros((self.buffer_size, self.num_envs), dtype=self.np_dtype)

        self.returns = None
        self.advantages = None

        self.samples_ready = False

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
            batch_size: int | None = None,
            shuffled: bool = True
    ) -> Generator[RolloutBufferSamples, None, None]:
        assert self.advantages is not None and self.returns is not None, \
            'Call compute_gae_and_returns before calling get_samples'

        num_samples = self.pos * self.num_envs

        if shuffled:
            indices = np.random.permutation(num_samples)
        else:
            indices = np.arange(num_samples)

        if batch_size is None:
            batch_size = num_samples

        if not self.samples_ready:
            array_names = [
                "observations",
                "actions",
                "action_log_probs",
                "value_estimates",
                "returns",
                "advantages",
            ]

            for array_name in array_names:
                # swapping and flattening creates a copy of the array. To prevent excess memory usage, we replace the
                # old arrays with the flattened ones, keeping memory usage the same.
                self.__dict__[array_name] = self.swap_and_flatten(self.__dict__[array_name])
            self.samples_ready = True

        for start_index in range(0, num_samples, batch_size):
            yield self.get_batch(indices[start_index:start_index + batch_size])

    def get_batch(self, batch_indices: np.ndarray):
        data = (
            self.observations[batch_indices],
            self.actions[batch_indices],
            self.action_log_probs[batch_indices],
            self.value_estimates[batch_indices],
            self.returns[batch_indices],
            self.advantages[batch_indices],
        )
        return RolloutBufferSamples(*self.all_to_torch(data))
