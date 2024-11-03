import numpy as np
import torch

from src.reinforcement_learning.core.buffers.replay.base_reservoir_replay_buffer import BaseReservoirReplayBuffer
from src.reinforcement_learning.core.type_aliases import NpObs, TensorObs
from src.torch_device import TorchDevice


class ReservoirReplayBuffer(BaseReservoirReplayBuffer):

    def __init__(
            self,
            total_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...],
            action_shape: tuple[int, ...],
            reward_scale: float,
            consider_truncated_as_done: bool = False,
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
            consider_truncated_as_done=consider_truncated_as_done,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
        )

        self.observations = np.zeros((total_size, *obs_shape), dtype=self.np_dtype)
        self.next_observations = np.zeros((total_size, *obs_shape), dtype=self.np_dtype)

    def _add_obs_at(self, indices: np.ndarray, observations: NpObs, next_observations: NpObs) -> None:
        self.observations[indices] = observations
        self.next_observations[indices] = next_observations

    def _get_batch_obs(self, batch_indices: np.ndarray) -> tuple[TensorObs, TensorObs]:
        observations = self.observations[batch_indices, :]
        next_observations = self.next_observations[batch_indices, :]
        return self.to_torch(observations), self.to_torch(next_observations)
