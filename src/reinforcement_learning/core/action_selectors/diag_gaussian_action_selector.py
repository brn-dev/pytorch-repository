from typing import Optional, Self

import torch
import torch.distributions as torchdist

from src.reinforcement_learning.core.action_selectors.continuous_action_selector import ContinuousActionSelector


# Inspired by
# https://github.com/DLR-RM/stable-baselines3/blob/285e01f64aa8ba4bd15aa339c45876d56ed0c3b4/stable_baselines3/common/distributions.py#L125
class DiagGaussianActionSelector(ContinuousActionSelector):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            std: float,
            std_learnable: bool
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            std=std,
            std_learnable=std_learnable
        )

        self.distribution: Optional[torchdist.Normal] = None

    def update_distribution_params(self, means: torch.Tensor, log_stds: torch.Tensor) -> Self:
        self.distribution = torchdist.Normal(loc=means, scale=torch.exp(log_stds))
        return self

    def sample(self) -> torch.Tensor:
        return self.distribution.rsample()

    def mode(self) -> torch.Tensor:
        return self.distribution.mean

    def actions_from_distribution_params(
            self,
            mean_actions: torch.Tensor,
            log_stds: torch.Tensor,
            deterministic: bool = False,
    ) -> torch.Tensor:
        self.update_distribution_params(mean_actions, log_stds)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_distribution_params(
            self,
            mean_actions: torch.Tensor,
            log_stds: torch.Tensor,
            *args, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_distribution_params(mean_actions, log_stds)
        log_prob = self.log_prob(actions)
        return actions, log_prob
