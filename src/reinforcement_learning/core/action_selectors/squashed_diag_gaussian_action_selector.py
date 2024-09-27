from typing import Optional, Self, Callable

import torch
from overrides import override
from torch import nn

from src.reinforcement_learning.core.action_selectors.action_selector import ActionNetInitialization
from src.reinforcement_learning.core.action_selectors.diag_gaussian_action_selector import DiagGaussianActionSelector
from src.reinforcement_learning.core.action_selectors.tanh_bijector import TanhBijector


# Inspired by
# https://github.com/DLR-RM/stable-baselines3/blob/285e01f64aa8ba4bd15aa339c45876d56ed0c3b4/stable_baselines3/common/distributions.py#L207
class SquashedDiagGaussianActionSelector(DiagGaussianActionSelector):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            std: float,
            std_learnable: bool,
            epsilon: float = 1e-6,
            action_net_initialization: ActionNetInitialization | None = None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            std=std,
            std_learnable=std_learnable,
            action_net_initialization=action_net_initialization,
        )

        self.epsilon = epsilon
        self._last_gaussian_actions: Optional[torch.Tensor] = None

    @override
    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        super().update_latent_features(latent_pi)
        return self

    @override
    def log_prob(self, actions: torch.Tensor, gaussian_actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if gaussian_actions is None:
            gaussian_actions = TanhBijector.inverse(actions)

        log_prob = super().log_prob(gaussian_actions)
        log_prob -= self.sum_action_dim(torch.log(1 - actions ** 2 + self.epsilon))

        return log_prob

    @override
    def entropy(self) -> torch.Tensor | None:
        return None

    @override
    def sample(self) -> torch.Tensor:
        self._last_gaussian_actions = super().sample()
        return torch.tanh(self._last_gaussian_actions)

    @override
    def mode(self) -> torch.Tensor:
        self._last_gaussian_actions = super().mode()
        return torch.tanh(self._last_gaussian_actions)

    @override
    def get_actions_with_log_probs(self, latent_pi: torch.Tensor, deterministic: bool = False):
        # get_actions calls sample() or mode(), both of which set _last_gaussian_actions
        # --> prevents squashing and unsquashing which can lead to numerical instability
        actions = self.update_latent_features(latent_pi).get_actions(deterministic=deterministic)
        log_probs = self.log_prob(actions, self._last_gaussian_actions)
        return actions, log_probs
