import abc
import math
from typing import Optional, Self

import torch
import torch.distributions as torchdist
from torch import nn

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector


class ContinuousActionSelector(ActionSelector, abc.ABC):

    log_stds: torch.Tensor

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
        )

        self.distribution: Optional[torchdist.Distribution] = None

        self.std_learnable = std_learnable

        self.set_log_stds_as_parameter(math.log(std))


    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        action_means = self.action_net(latent_pi)
        return self.update_distribution_params(action_means, self.log_stds)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return sum_independent_dims(self.distribution.log_prob(actions))

    def entropy(self) -> torch.Tensor | None:
        return sum_independent_dims(self.distribution.entropy())

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


def sum_independent_dims(tensor: torch.Tensor) -> torch.Tensor:
    """
    Continuous actions are usually considered to be independent,
    so we can sum components of the ``log_prob`` or the entropy.

    :param tensor: shape: (n_batch, n_actions) or (n_batch,)
    :return: shape: (n_batch,) for (n_batch, n_actions) input, scalar for (n_batch,) input
    """
    if len(tensor.shape) > 1:
        tensor = tensor.sum(dim=1)
    else:
        tensor = tensor.sum()
    return tensor
