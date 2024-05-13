from typing import Optional, Self

import torch
from overrides import override

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
            sum_action_dim: bool = False,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            std=std,
            std_learnable=std_learnable,
            sum_action_dim=sum_action_dim
        )

        self.epsilon = epsilon
        self.last_unsquashed_actions: Optional[torch.Tensor] = None

    @override
    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        super().update_latent_features(latent_pi)
        return self

    @override
    def log_prob(self, actions: torch.Tensor, unsquashed_action: Optional[torch.Tensor] = None) -> torch.Tensor:
        if unsquashed_action is None:
            unsquashed_action = TanhBijector.inverse(actions)

        log_prob = super().log_prob(unsquashed_action)
        log_prob -= self.sum_action_dim(torch.log(1 - actions ** 2 + self.epsilon))

        return log_prob

    @override
    def entropy(self) -> torch.Tensor | None:
        return None

    @override
    def sample(self) -> torch.Tensor:
        self.last_unsquashed_actions = super().sample()
        return torch.tanh(self.last_unsquashed_actions)

    @override
    def mode(self) -> torch.Tensor:
        self.last_unsquashed_actions = super().mode()
        return torch.tanh(self.last_unsquashed_actions)

    @override
    def log_prob_from_distribution_params(
            self,
            mean_actions: torch.Tensor,
            log_stds: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = self.actions_from_distribution_params(mean_actions, log_stds)
        log_prob = self.log_prob(action, self.last_unsquashed_actions)
        return action, log_prob
