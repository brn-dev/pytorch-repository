import abc
from typing import Self

import torch

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector, ActionNetInitialization
from src.tags import Tags


class DiscreteActionSelector(ActionSelector, abc.ABC):

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

    def collect_tags(self) -> Tags:
        return self.combine_tags(super().collect_tags(), ['Discrete Action Space'])

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        action_logits = self.action_net(latent_pi)
        return self.update_distribution_params(action_logits)
