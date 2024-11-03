import functools
import math
from typing import Type, Any, Callable

import numpy as np
import torch
from gymnasium import Env

from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.buffers.replay.base_ring_replay_buffer import ReplayBufferSamples
from src.reinforcement_learning.core.buffers.replay.ring_replay_buffer import RingReplayBuffer
from src.reinforcement_learning.core.buffers.replay.reservoir_replay_buffer import ReservoirReplayBuffer
from src.reinforcement_learning.core.type_aliases import NpObs
from src.reinforcement_learning.gym.env_analysis import get_num_envs, get_obs_shape, get_action_shape
from src.repr_utils import func_repr
from src.torch_device import TorchDevice

ReservoirSamplingRatioFunc = Callable[['ReplayWithReservoirBuffer', int, int], float]


class RingWithReservoirReplayBuffer(RingReplayBuffer):

    def __init__(
            self,
            reservoir: ReservoirReplayBuffer,
            replay_step_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...],
            action_shape: tuple[int, ...],
            reward_scale: float,
            reservoir_sampling_ratio: float | ReservoirSamplingRatioFunc = None,
            min_reservoir_size: int = 1,
            consider_truncated_as_done: bool = False,
            torch_device: TorchDevice = 'cpu',
            torch_dtype: torch.dtype = torch.float32,
            np_dtype: np.dtype = np.float32,
    ):
        super().__init__(
            step_size=replay_step_size,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_scale=reward_scale,
            optimize_memory_usage=False,  # TODO
            consider_truncated_as_done=consider_truncated_as_done,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
        )

        self.reservoir = reservoir

        reservoir_total_size = self.reservoir.total_size
        replay_total_size = self.step_size * self.num_envs
        # May be used in custom ratio functions
        self.reservoir_total_size_ratio = reservoir_total_size / (replay_total_size + reservoir_total_size)

        if reservoir_sampling_ratio is not None:
            self.reservoir_sampling_ratio = reservoir_sampling_ratio
        else:
            self.reservoir_sampling_ratio = functools.partial(self._default_reservoir_sampling_ratio, self)

        self.min_reservoir_size = min_reservoir_size

        assert self.consider_truncated_as_done == self.reservoir.consider_truncated_as_done

    # Static so it can be used inside custom ratio functions
    @staticmethod
    def _default_reservoir_sampling_ratio(self: 'RingWithReservoirReplayBuffer', replay_count: int, reservoir_count: int):
        return reservoir_count / (replay_count + reservoir_count)

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'reservoir': self.reservoir.collect_hyper_parameters(),
            'reservoir_sampling_ratio':
                func_repr(self.reservoir_sampling_ratio)
                if isinstance(self.reservoir_sampling_ratio, Callable)
                else self.reservoir_sampling_ratio,
            'min_reservoir_size': self.min_reservoir_size
        })

    def add(
            self,
            observations: NpObs,
            next_observations: NpObs,
            actions: np.ndarray,
            rewards: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray,
    ) -> None:
        if self.full:
            self.reservoir.add(
                observations=self.observations[self.pos],
                next_observations=self.next_observations[self.pos],
                actions=self.actions[self.pos],
                rewards=self.unscale_rewards(self.rewards[self.pos]),
                terminated=self.terminated[self.pos],
                truncated=self.truncated[self.pos]
            )

        super().add(
            observations=observations,
            next_observations=next_observations,
            actions=actions,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
        )

    def sample(
            self,
            batch_size: int
    ) -> ReplayBufferSamples:
        reservoir_count = self.reservoir.count

        if reservoir_count == 0:
            return super().sample(batch_size)

        if isinstance(self.reservoir_sampling_ratio, float):
            reservoir_sampling_ratio = self.reservoir_sampling_ratio
        else:
            reservoir_sampling_ratio = self.reservoir_sampling_ratio(self.count, reservoir_count)

        remainder, num_reservoir_samples = math.modf(reservoir_sampling_ratio * batch_size)
        num_reservoir_samples = int(num_reservoir_samples)

        if self.rng.random() < remainder:
            num_reservoir_samples += 1

        if reservoir_count < self.min_reservoir_size if self.min_reservoir_size is not None else num_reservoir_samples:
            return super().sample(batch_size)

        num_replay_samples = batch_size - num_reservoir_samples

        replay_samples = super().sample(num_replay_samples)
        reservoir_samples = self.reservoir.sample(num_reservoir_samples)

        with torch.no_grad():
            return ReplayBufferSamples(*[
                torch.cat(samples, dim=0) for samples in zip(replay_samples, reservoir_samples)
            ])

    # noinspection PyMethodOverriding
    @classmethod
    def for_env(
            cls,
            env: Env,
            buffer_size: int,
            reservoir_total_size: int,
            reward_scale: float,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype = torch.float32,
            np_dtype: np.dtype = np.float32,
            reservoir_type: Type[ReservoirReplayBuffer] = ReservoirReplayBuffer,
            reservoir_kwargs: dict[str, Any] = None,
            **buffer_kwargs
    ):
        reservoir_buffer = reservoir_type.for_env(
            env=env,
            buffer_size=reservoir_total_size,
            reward_scale=reward_scale,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
            **(reservoir_kwargs or {}),
        )

        num_envs = get_num_envs(env)

        obs_shape = get_obs_shape(env)
        action_shape = get_action_shape(env)

        # noinspection PyArgumentList
        return cls(
            reservoir=reservoir_buffer,
            replay_step_size=buffer_size,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_scale=reward_scale,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
            **buffer_kwargs
        )








