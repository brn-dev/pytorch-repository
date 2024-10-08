from typing import Generic, TypeVar

import numpy as np
import torch

from src.reinforcement_learning.core.buffers.replay.base_replay_buffer import ReplayBufferSamples
from src.reinforcement_learning.core.buffers.replay.replay_buffer import ReplayBuffer
from src.reinforcement_learning.core.buffers.resevoir.reservoir_buffer import ReservoirBuffer
from src.reinforcement_learning.core.type_aliases import NpObs
from src.torch_device import TorchDevice


ReservoirBuf = TypeVar('ReservoirBuf', bound=ReservoirBuffer)


class ReplayWithReservoirBuffer(ReplayBuffer, Generic[ReservoirBuf]):

    def __init__(
            self,
            reservoir: ReservoirBuf,
            reservoir_ratio: float,
            step_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...],
            action_shape: tuple[int, ...],
            reward_scale: float,
            min_reservoir_size: int = None,
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
            optimize_memory_usage=False,  # TODO
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
        )

        self.reservoir = reservoir
        self.reservoir_ratio = reservoir_ratio
        self.min_reservoir_size = min_reservoir_size

    def add(
            self,
            observations: NpObs,
            next_observations: NpObs,
            actions: np.ndarray,
            rewards: np.ndarray,
            dones: np.ndarray,
    ) -> None:
        if self.full:
            self.reservoir.add(
                observations=self.observations[self.pos],
                next_observations=self.next_observations[self.pos],
                actions=self.actions[self.pos],
                rewards=self.rewards[self.pos],
                dones=self.dones[self.pos],
            )

        super().add(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            dones=dones,
        )

    def sample(
            self,
            batch_size: int
    ) -> ReplayBufferSamples:
        num_reservoir_samples = round(self.reservoir_ratio * batch_size)

        if self.reservoir.size < (self.min_reservoir_size or num_reservoir_samples):
            return super().sample(batch_size)

        num_replay_samples = batch_size - num_reservoir_samples

        replay_samples = super().sample(num_replay_samples)
        reservoir_samples = self.reservoir.sample(num_reservoir_samples)

        with torch.no_grad():
            return ReplayBufferSamples(*[
                torch.cat(samples, dim=0) for samples in zip(replay_samples, reservoir_samples)
            ])




