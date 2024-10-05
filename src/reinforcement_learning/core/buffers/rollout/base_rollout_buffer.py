import abc
from typing import Generator

import numpy as np
import torch

from src.reinforcement_learning.core.buffers.base_buffer import BaseBuffer, BufferSamples
from src.reinforcement_learning.core.generalized_advantage_estimate import compute_returns_and_gae
from src.reinforcement_learning.core.normalization import NormalizationType
from src.reinforcement_learning.core.type_aliases import NpObs


# TODO
class BaseRolloutBuffer(BaseBuffer[BufferSamples], abc.ABC):

    rewards: np.ndarray
    episode_starts: np.ndarray
    actions: np.ndarray
    action_log_probs: np.ndarray
    value_estimates: np.ndarray
    returns: np.ndarray | None
    advantages: np.ndarray | None

    @abc.abstractmethod
    def add(
            self,
            observations: NpObs,
            rewards: np.ndarray,
            episode_starts: np.ndarray,
            actions: torch.Tensor,
            action_log_probs: torch.Tensor,
            value_estimates: torch.Tensor,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def get_samples(
            self,
            batch_size: int | None = None,
    ) -> Generator[BufferSamples, None, None]:
        raise NotImplementedError

    def compute_returns_and_gae(
            self,
            last_values: torch.Tensor,
            last_episode_starts: np.ndarray,
            gamma: float,
            gae_lambda: float,
            normalize_rewards: NormalizationType | None,
            normalize_advantages: NormalizationType | None,
            compensate_for_reward_scaling: bool = True,
    ) -> None:
        assert self.returns is None and self.advantages is None, 'returns and gae already computed'

        last_values = last_values.squeeze(-1).detach().clone().cpu().numpy()

        rewards = self.rewards[:self.pos]
        if compensate_for_reward_scaling:
            rewards = self.unscale_rewards(rewards)

        self.returns, self.advantages = compute_returns_and_gae(
            value_estimates=self.value_estimates[:self.pos],
            rewards=rewards,
            episode_starts=self.episode_starts[:self.pos],
            last_values=last_values,
            last_episode_starts=last_episode_starts,
            gamma=gamma,
            gae_lambda=gae_lambda,
            normalize_rewards=normalize_rewards,
            normalize_advantages=normalize_advantages
        )

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
