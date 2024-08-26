from typing import Generator

import numpy as np
import torch

from src.reinforcement_learning.core.buffers.base_buffer import BaseBuffer
from src.reinforcement_learning.core.buffers.rollout.rollout_buffer_samples import RolloutBufferSamples
from src.reinforcement_learning.core.generalized_advantage_estimate import compute_returns_and_gae
from src.reinforcement_learning.core.normalization import NormalizationType
from src.torch_device import TorchDevice


class RolloutBuffer(BaseBuffer[RolloutBufferSamples]):

    observations: np.ndarray
    rewards: np.ndarray
    episode_starts: np.ndarray
    actions: np.ndarray
    action_log_probs: np.ndarray
    value_estimates: np.ndarray
    returns: np.ndarray | None
    advantages: np.ndarray | None
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

    def compute_returns_and_gae(
            self,
            last_values: torch.Tensor,
            last_episode_starts: np.ndarray,
            gamma: float,
            gae_lambda: float,
            normalize_rewards: NormalizationType | None,
            normalize_advantages: NormalizationType | None,
    ) -> None:
        assert self.returns is None and self.advantages is None, 'returns and gae already computed'

        last_values = last_values.squeeze(-1).detach().clone().cpu().numpy()

        self.returns, self.advantages = compute_returns_and_gae(
            value_estimates=self.value_estimates[:self.pos],
            rewards=self.rewards[:self.pos],
            episode_starts=self.episode_starts[:self.pos],
            last_values=last_values,
            last_episode_starts=last_episode_starts,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_rewards=normalize_rewards,
            normalize_advantages=normalize_advantages
        )

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

    def compute_critic_explained_variance(self, returns: np.ndarray | None = None):
        assert returns is not None or self.returns is not None, \
            'if returns is not provided as parameter, compute_returns_and_gae has to be called beforehand'

        if returns is None:
            returns = self.returns
        returns = returns.ravel()

        return_variance = np.var(returns)

        if return_variance == 0:
            return np.nan

        value_predictions = self.value_estimates.ravel()
        return 1 - np.var(returns - value_predictions) / return_variance

