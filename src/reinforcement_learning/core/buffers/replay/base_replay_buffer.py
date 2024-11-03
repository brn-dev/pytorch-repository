import abc
from typing import NamedTuple

import numpy as np
import torch
from gymnasium import Env

from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.buffers.base_buffer import BaseBuffer
from src.reinforcement_learning.core.type_aliases import ShapeDict, NpObs, TensorObs
from src.torch_device import TorchDevice


class ReplayBufferSamples(NamedTuple):
    observations: TensorObs
    actions: torch.Tensor
    next_observations: TensorObs
    dones: torch.Tensor
    rewards: torch.Tensor


class BaseReplayBuffer(BaseBuffer[ReplayBufferSamples], abc.ABC):

    def __init__(
            self,
            step_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...] | ShapeDict,
            action_shape: tuple[int, ...],
            reward_scale: float,
            consider_truncated_as_done: bool,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype,
            np_dtype: np.dtype,
            rng_seed: int = None,
    ):
        super().__init__(
            step_size=step_size,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            reward_scale=reward_scale,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
        )

        self.consider_truncated_as_done = consider_truncated_as_done
        self.rng = np.random.default_rng(rng_seed)

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'consider_truncated_as_done': self.consider_truncated_as_done,
        })

    @abc.abstractmethod
    def sample(self, batch_size: int) -> ReplayBufferSamples:
        raise NotImplementedError

    def tail_indices(self, tail_length: int):
        if tail_length > self.step_count:
            tail_length = self.step_count

        if not self.full or self.pos >= tail_length:
            return np.arange(self.pos - tail_length, self.pos)

        return np.concatenate((
            np.arange(self.step_count - tail_length + self.pos, self.step_count),
            np.arange(self.pos)
        ))

    def _compute_most_recent_episode_scores(
            self,
            rewards: np.ndarray,
            terminated: np.ndarray,
            truncated: np.ndarray,
            n_episodes: int,
            compensate_for_reward_scaling: bool,
            consider_truncated_as_done: bool,
    ):
        if consider_truncated_as_done is None:
            consider_truncated_as_done = self.consider_truncated_as_done

        whole_episode = np.zeros((self.num_envs,), dtype=bool)

        running_sum = np.zeros((self.num_envs,), dtype=float)
        episode_scores: list[float] = []

        for step_index in reversed(self.tail_indices(self.step_count)):
            if consider_truncated_as_done:
                step_dones = np.logical_or(terminated[step_index], truncated[step_index])
            else:
                step_dones = terminated[step_index]

            episode_scores.extend(running_sum[np.logical_and(step_dones, whole_episode)])

            if len(episode_scores) >= n_episodes:
                break

            whole_episode[step_dones] = True
            running_sum[step_dones] = 0.0

            step_rewards = rewards[step_index]
            if compensate_for_reward_scaling:
                step_rewards = self.unscale_rewards(step_rewards)

            running_sum += step_rewards

        return np.array(episode_scores)

    # noinspection PyMethodOverriding
    @classmethod
    def for_env(
            cls,
            env: Env,
            buffer_size: int,
            reward_scale: float,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype = torch.float32,
            np_dtype: np.dtype = np.float32,
            **buffer_kwargs
    ):
        return super().for_env(
            env=env,
            buffer_size=buffer_size,
            reward_scale=reward_scale,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            **buffer_kwargs
        )
