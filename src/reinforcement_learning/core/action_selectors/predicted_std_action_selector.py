from typing import Optional, Self, Callable

import torch
import torch.distributions as torchdist
from overrides import override
from torch import nn

from src.function_types import TorchTensorTransformation
from src.reinforcement_learning.core.action_selectors.action_selector import ActionNetInitialization
from src.reinforcement_learning.core.action_selectors.continuous_action_selector import ContinuousActionSelector
from src.reinforcement_learning.core.action_selectors.tanh_bijector import TanhBijector

LogStdNetInitialization = ActionNetInitialization

class PredictedStdActionSelector(ContinuousActionSelector):

    output_bijector: Optional[TanhBijector]
    initial_log_std: float

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            initial_std: float,
            squash_output: bool = False,
            epsilon: float = 1e-6,
            sum_action_dim: bool = False,
            action_net_initialization: ActionNetInitialization | None = None,
            log_std_net_initialization: LogStdNetInitialization | None = None,
            log_std_activation: TorchTensorTransformation = lambda x: x,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            std=initial_std,
            std_learnable=True,
            sum_action_dim=sum_action_dim,
            action_net_initialization=action_net_initialization,
        )

        self.log_std_net = nn.Linear(latent_dim, action_dim)
        if log_std_net_initialization is not None:
            log_std_net_initialization(self.log_std_net)
        self.log_std_activation = log_std_activation

        if squash_output:
            self.output_bijector = TanhBijector(epsilon)
        else:
            self.output_bijector = None

        self.epsilon = epsilon
        self.distribution: Optional[torchdist.Normal] = None

    @override
    def set_log_stds_as_parameter(self, initial_log_std: float) -> None:
        self.initial_log_std = initial_log_std

    @override
    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        mean_actions = self.action_net(latent_pi)
        log_stds = self.log_std_net(latent_pi) + self.initial_log_std
        return self.update_distribution_params(mean_actions, log_stds)

    def update_distribution_params(
            self,
            mean_actions: torch.Tensor,
            log_stds: torch.Tensor,
    ) -> Self:
        self.distribution = torchdist.Normal(mean_actions, log_stds.exp())
        return self

    def sample(self) -> torch.Tensor:
        actions = self.distribution.rsample()
        if self.output_bijector is not None:
            return self.output_bijector.forward(actions)
        return actions

    def mode(self) -> torch.Tensor:
        actions = self.distribution.mean
        if self.output_bijector is not None:
            return self.output_bijector.forward(actions)
        return actions

    @override
    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        if self.output_bijector is not None:
            unsquashed_actions = self.output_bijector.inverse(actions)
        else:
            unsquashed_actions = actions

        log_prob = self.distribution.log_prob(unsquashed_actions)
        log_prob = self.sum_action_dim(log_prob)

        if self.output_bijector is not None:
            log_prob -= self.sum_action_dim(self.output_bijector.log_prob_correction(unsquashed_actions))
        return log_prob

    @override
    def entropy(self) -> torch.Tensor | None:
        if self.output_bijector is not None:
            return None
        return self.sum_action_dim(self.distribution.entropy())

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_distribution_params(mean_actions, log_stds)
        log_prob = self.log_prob(actions)
        return actions, log_prob
