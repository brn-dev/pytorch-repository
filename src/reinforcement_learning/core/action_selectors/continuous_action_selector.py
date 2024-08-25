import abc
from typing import Optional, Self

import torch
import torch.distributions as torchdist

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector, ActionNetInitialization


class ContinuousActionSelector(ActionSelector, abc.ABC):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )

        self.distribution: Optional[torchdist.Distribution] = None

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        action_means = self.action_net(latent_pi)
        return self.update_distribution_params(action_means, self.log_stds)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.sum_action_dim(self.distribution.log_prob(actions))

    def entropy(self) -> torch.Tensor | None:
        return self.sum_action_dim(self.distribution.entropy())

    @staticmethod
    def sum_action_dim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sum(dim=-1)
