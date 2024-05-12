from typing import Optional, Self

import torch
import torch.distributions as torchdist

from src.reinforcement_learning.core.action_selectors.discrete_action_selector import DiscreteActionSelector


# Inspired by
# https://github.com/DLR-RM/stable-baselines3/blob/285e01f64aa8ba4bd15aa339c45876d56ed0c3b4/stable_baselines3/common/distributions.py#L263
class CategoricalActionSelector(DiscreteActionSelector):

    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
        )

        self.distribution: Optional[torchdist.Categorical] = None

    def update_distribution_params(self, action_logits: torch.Tensor) -> Self:
        self.distribution = torchdist.Categorical(logits=action_logits)
        return self

    def sample(self) -> torch.Tensor:
        return self.distribution.sample()

    def mode(self) -> torch.Tensor:
        return torch.argmax(self.distribution.probs, dim=1)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> torch.Tensor | None:
        return self.distribution.entropy()
