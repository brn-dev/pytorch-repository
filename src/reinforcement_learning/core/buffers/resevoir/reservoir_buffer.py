import numpy as np
import torch

from src.reinforcement_learning.core.buffers.resevoir.base_reservoir_buffer import BaseReservoirBuffer, \
    ReservoirBufferSamples
from src.reinforcement_learning.core.type_aliases import NpObs, ShapeDict
from src.torch_device import TorchDevice


class ReservoirBuffer(BaseReservoirBuffer):

    def __init__(
            self,
            total_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...] | ShapeDict,
            action_shape: tuple[int, ...],
            reward_scale: float,
            torch_device: TorchDevice = 'cpu',
            torch_dtype: torch.dtype = torch.float32,
            np_dtype: np.dtype = np.float32,
    ):
        super().__init__(
            total_size=total_size,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_scale=reward_scale,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
        )

        self.observations = np.zeros((total_size, *obs_shape), dtype=self.np_dtype)
        self.next_observations = np.zeros((total_size, *obs_shape), dtype=self.np_dtype)

    def _add_obs_at(self, indices: np.ndarray, observations: NpObs, next_observations: NpObs) -> None:
        self.observations[indices] = observations
        self.next_observations[indices] = next_observations

    def _get_batch(self, batch_indices: np.ndarray) -> ReservoirBufferSamples:
        data = (
            self.observations[batch_indices, :],
            self.actions[batch_indices, :],
            self.next_observations[batch_indices, :],
            self.dones[batch_indices].reshape(-1, 1),
            self.rewards[batch_indices].reshape(-1, 1),
        )
        return ReservoirBufferSamples(*self.all_to_torch(data))
