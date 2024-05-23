import abc
import math
from typing import Optional, Self, Callable

import torch
import torch.distributions as torchdist
from torch import nn

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector, ActionNetInitialization


class ContinuousActionSelector(ActionSelector, abc.ABC):

    log_stds: torch.Tensor

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            std: float,
            std_learnable: bool,
            sum_action_dim: bool,
            action_net_initialization: ActionNetInitialization | None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )

        self.distribution: Optional[torchdist.Distribution] = None

        self.std_learnable = std_learnable
        self.set_log_stds_as_parameter(math.log(std))

        self._sum_action_dim = sum_action_dim

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        action_means = self.action_net(latent_pi)
        return self.update_distribution_params(action_means, self.log_stds)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.sum_action_dim(self.distribution.log_prob(actions))

    def entropy(self) -> torch.Tensor | None:
        return self.sum_action_dim(self.distribution.entropy())

    def set_log_std(self, log_std: float) -> None:
        if self.std_learnable:
            # TODO: proper logging
            print('Warning: setting std while std is learnable!')

        self.set_log_stds_as_parameter(log_std)

    def set_log_stds_as_parameter(self, initial_log_std: float) -> None:
        self.log_stds = nn.Parameter(
            torch.ones((self.action_dim,)) * initial_log_std,
            requires_grad=self.std_learnable
        )

    def sum_action_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._sum_action_dim:
            return tensor.sum(dim=-1)
        else:
            return tensor
