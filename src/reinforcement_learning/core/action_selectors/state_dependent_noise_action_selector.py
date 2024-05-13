from typing import Optional, Self

import torch
import torch.distributions as torchdist
from overrides import override
from torch import nn

from src.reinforcement_learning.core.action_selectors.continuous_action_selector import ContinuousActionSelector
from src.reinforcement_learning.core.action_selectors.tanh_bijector import TanhBijector


# Inspired by
# https://github.com/DLR-RM/stable-baselines3/blob/285e01f64aa8ba4bd15aa339c45876d56ed0c3b4/stable_baselines3/common/distributions.py#L421
class StateDependentNoiseActionSelector(ContinuousActionSelector):

    output_bijector: Optional[TanhBijector]
    noise_weights_dist: torchdist.Normal
    exploration_noise_mat: torch.Tensor
    exploration_noise_matrices: torch.Tensor

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            initial_std: float,
            use_full_stds: bool = True,
            use_stds_expln: bool = False,
            squash_output: bool = False,
            learn_sde_features: bool = True,
            epsilon: float = 1e-6,
            sum_action_dim: bool = False,
    ):
        self.use_full_stds = use_full_stds

        self.distribution: Optional[torchdist.Normal] = None
        self.use_stds_expln = use_stds_expln
        self.learn_sde_features = learn_sde_features

        self.latent_pi: Optional[torch.Tensor] = None

        self.epsilon = epsilon

        if squash_output:
            self.output_bijector = TanhBijector(epsilon)
        else:
            self.output_bijector = None

        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            std=initial_std,
            std_learnable=True,
            sum_action_dim=sum_action_dim,
        )

    @override
    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        mean_actions = self.action_net(latent_pi)
        return self.update_distribution_params(mean_actions, self.log_stds, latent_pi)

    def update_distribution_params(
            self,
            mean_actions: torch.Tensor,
            log_stds: torch.Tensor,
            latent_pi: torch.Tensor
    ) -> Self:
        self.latent_pi = latent_pi
        variance = torch.mm(self.latent_pi ** 2, self.get_stds(self.log_stds) ** 2)
        self.distribution = torchdist.Normal(mean_actions, torch.sqrt(variance + self.epsilon))
        return self

    def sample(self) -> torch.Tensor:
        noise = self.get_noise(self.latent_pi)
        actions = self.distribution.mean + noise
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
            log_prob -= torch.sum(self.output_bijector.log_prob_correction(unsquashed_actions), dim=1)
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
            latent_pi: torch.Tensor,
            deterministic: bool = False,
    ) -> torch.Tensor:
        self.update_distribution_params(mean_actions, log_stds, latent_pi)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_distribution_params(
            self,
            mean_actions: torch.Tensor,
            log_stds: torch.Tensor,
            latent_pi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = self.actions_from_distribution_params(mean_actions, log_stds, latent_pi)
        log_prob = self.log_prob(actions)
        return actions, log_prob

    def get_stds(self, log_stds: torch.Tensor) -> torch.Tensor:
        if self.use_stds_expln:
            below_threshold = torch.exp(log_stds) * (log_stds <= 0)
            safe_log_stds = log_stds * (log_stds > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_stds) + 1.0) * (log_stds > 0)
            stds = below_threshold + above_threshold
        else:
            stds = torch.exp(log_stds)

        if self.use_full_stds:
            return stds

        return torch.ones((self.latent_dim, self.action_dim)).to(log_stds.device) * stds

    def sample_noise_weights(self, batch_size: int = 1):
        stds = self.get_stds(self.log_stds)
        self.noise_weights_dist = torchdist.Normal(loc=torch.zeros_like(stds), scale=stds)
        self.exploration_noise_mat = self.noise_weights_dist.rsample()
        self.exploration_noise_matrices = self.noise_weights_dist.rsample(torch.Size((batch_size,)))

    def get_noise(self, latent_pi: torch.Tensor):
        latent_pi = latent_pi if self.learn_sde_features else latent_pi.detach()

        if len(latent_pi) == 1:  # or len(latent_pi) != len(self.exploration_noise_matrices): # TODO: is this necessary?
            return torch.mm(latent_pi, self.exploration_noise_mat)

        latent_pi = latent_pi.unsqueeze(dim=1)
        print(f'{latent_pi.shape = }')
        print(f'{self.exploration_noise_matrices.shape = }')
        noise = torch.bmm(latent_pi, self.exploration_noise_matrices)
        return noise.squeeze(dim=1)

    @override
    def set_log_stds_as_parameter(self, initial_log_std: float) -> None:
        if self.use_full_stds:
            log_stds = torch.ones((self.latent_dim, self.action_dim))
        else:
            log_stds = torch.ones((self.latent_dim, 1))

        self.log_stds = nn.Parameter(log_stds, requires_grad=self.std_learnable)
        self.sample_noise_weights()
