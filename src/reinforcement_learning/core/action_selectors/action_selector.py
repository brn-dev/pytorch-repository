import abc
import math
from typing import TypeVar, Callable, Optional

import torch
import torch.distributions as torchdist
from torch import nn

SelfActionDistribution = TypeVar('SelfActionDistribution', bound='ActionDistribution')

ActionNetInitialization = Callable[[nn.Linear], None]


class ActionSelector(nn.Module, abc.ABC):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization | None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim

        if latent_dim == action_dim:
            self.action_net = nn.Identity()
        else:
            self.action_net = nn.Linear(latent_dim, action_dim)

            if action_net_initialization is not None:
                action_net_initialization(self.action_net)

        self.distribution: Optional[torchdist.Distribution] = None

    @abc.abstractmethod
    def update_latent_features(self: SelfActionDistribution, latent_pi: torch.Tensor) -> SelfActionDistribution:
        raise NotImplementedError

    @abc.abstractmethod
    def update_distribution_params(self: SelfActionDistribution, *args, **kwargs) -> SelfActionDistribution:
        raise NotImplementedError

    @abc.abstractmethod
    def sample(self) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def mode(self) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abc.abstractmethod
    def entropy(self) -> torch.Tensor | None:
        raise NotImplementedError

    def get_actions(self, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mode()
        return self.sample()

    def get_actions_with_log_probs(self, latent_pi: torch.Tensor, deterministic: bool = False):
        actions = self.update_latent_features(latent_pi).get_actions(deterministic=deterministic)
        log_probs = self.log_prob(actions)
        return actions, log_probs
