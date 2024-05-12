import abc
from typing import Optional, Self

import torch
import torch.distributions as torchdist

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector


class DiscreteActionSelector(ActionSelector, abc.ABC):

    def __init__(self, latent_dim: int, action_dim: int):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
        )

        self.distribution: Optional[torchdist.Categorical] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        action_logits = self.action_net(latent_pi)
        return self.update_distribution_params(action_logits)

    def actions_from_distribution_params(
            self,
            action_logits: torch.Tensor,
            deterministic: bool = False
    ) -> torch.Tensor:
        self.update_distribution_params(action_logits)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_distribution_params(self, action_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_distribution_params(action_logits)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def set_log_std(self, log_std: float) -> None:
        print('Warning: trying to set std on a categorical action distribution')
