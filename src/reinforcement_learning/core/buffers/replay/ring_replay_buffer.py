import numpy as np
import torch
from overrides import override

from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.buffers.replay.base_ring_replay_buffer import BaseRingReplayBuffer, ReplayBufferSamples
from src.reinforcement_learning.core.type_aliases import TensorObs
from src.torch_device import TorchDevice


class RingReplayBuffer(BaseRingReplayBuffer):

    def __init__(
            self,
            step_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...],
            action_shape: tuple[int, ...],
            reward_scale: float,
            optimize_memory_usage: bool = False,
            consider_truncated_as_done: bool = False,
            torch_device: TorchDevice = 'cpu',
            torch_dtype: torch.dtype = torch.float32,
            np_dtype: np.dtype = np.float32,
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

        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.step_size, self.num_envs, *self.obs_shape), dtype=self.np_dtype)
        if not optimize_memory_usage:
            self.next_observations = np.zeros((self.step_size, self.num_envs, *obs_shape), dtype=self.np_dtype)

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'optimize_memory_usage': self.optimize_memory_usage,
        })

    def _add_obs(
            self,
            observations: np.ndarray,
            next_observations: np.ndarray,
    ) -> None:
        self.observations[self.pos] = observations

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.step_size] = next_observations
        else:
            self.next_observations[self.pos] = next_observations

    @override
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        env_indices = self.rng.choice(self.num_envs, size=batch_size)

        if not self.optimize_memory_usage:
            step_indices = self.rng.choice(self.step_count, batch_size)
            return self._get_batch(step_indices, env_indices)

        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            step_indices = (self.rng.choice(self.step_size - 1, batch_size) + self.pos + 1) % self.step_size
        else:
            step_indices = self.rng.choice(self.pos, batch_size)

        return self._get_batch(step_indices, env_indices)

    @override
    def _get_batch_obs(self, step_indices: np.ndarray, env_indices: np.ndarray) -> tuple[TensorObs, TensorObs]:
        obs = self.to_torch(self.observations[step_indices, env_indices, :])

        if self.optimize_memory_usage:
            next_obs = self.observations[(step_indices + 1) % self.step_size, env_indices, :]
        else:
            next_obs = self.next_observations[step_indices, env_indices, :]
        next_obs = self.to_torch(next_obs)

        return obs, next_obs
