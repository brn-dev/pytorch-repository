import math
from typing import Optional, Self, Callable

import torch
import torch.distributions as torchdist
from overrides import override
from torch import nn

from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.action_selectors.action_selector import ActionNetInitialization
from src.reinforcement_learning.core.action_selectors.continuous_action_selector import ContinuousActionSelector
from src.reinforcement_learning.core.action_selectors.tanh_bijector import TanhBijector


# Inspired by
# https://github.com/DLR-RM/stable-baselines3/blob/285e01f64aa8ba4bd15aa339c45876d56ed0c3b4/stable_baselines3/common/distributions.py#L421
class StateDependentNoiseActionSelector(ContinuousActionSelector):

    output_bijector: Optional[TanhBijector]
    exploration_noise_dist: torchdist.Normal
    exploration_noise_matrices: torch.Tensor

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            initial_std: float,
            use_individual_action_stds: bool = True,
            use_stds_expln: bool = False,
            squash_output: bool = False,
            learn_sde_features: bool = True,
            clip_mean: float = 0.0,
            epsilon: float = 1e-6,
            action_net_initialization: ActionNetInitialization | None = None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )
        self.initial_std = initial_std

        self.use_individual_action_stds = use_individual_action_stds
        if self.use_individual_action_stds:
            log_stds = torch.ones((self.latent_dim, self.action_dim))
        else:
            log_stds = torch.ones((self.latent_dim, 1))
        self.log_stds = nn.Parameter(log_stds * math.log(initial_std), requires_grad=True)
        self.use_stds_expln = use_stds_expln
        self.sample_exploration_noise()

        self.distribution: Optional[torchdist.Normal] = None
        self.learn_sde_features = learn_sde_features

        self.latent_pi: Optional[torch.Tensor] = None

        assert clip_mean >= 0.0
        self.clip_mean = clip_mean if clip_mean > 0 else None

        self.epsilon = epsilon

        self.squash_output = squash_output
        if squash_output:
            self.output_bijector = TanhBijector(epsilon)
        else:
            self.output_bijector = None

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'initial_std': self.initial_std,
            'use_individual_action_stds': self.use_individual_action_stds,
            'use_stds_expln': self.use_stds_expln,
            'squash_output': self.squash_output,
            'learn_sde_features': self.learn_sde_features,
            'clip_mean': 0 if self.clip_mean is None else self.clip_mean,
            'epsilon': self.epsilon,
        })

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
        if self.latent_pi.dim() == 2:
            variance = torch.mm(self.latent_pi ** 2, self.get_stds(self.log_stds) ** 2)
        else:
            batch_shape = self.latent_pi.shape[:-1]
            flat_latent_pi = torch.flatten(latent_pi, end_dim=-2)
            variance = torch.mm(flat_latent_pi ** 2, self.get_stds(self.log_stds) ** 2)
            variance = variance.reshape(batch_shape + (self.action_dim,))

        if self.clip_mean is not None:
            mean_actions = nn.functional.hardtanh(mean_actions, -self.clip_mean, self.clip_mean)

        self.distribution = torchdist.Normal(mean_actions, torch.sqrt(variance + self.epsilon))
        return self

    # TODO: save last_gaussian_actions like in the other squashing action selectors?
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
            gaussian_actions = self.output_bijector.inverse(actions)
        else:
            gaussian_actions = actions

        log_prob = self.distribution.log_prob(gaussian_actions)
        log_prob = self.sum_action_dim(log_prob)

        if self.output_bijector is not None:
            log_prob -= self.sum_action_dim(self.output_bijector.log_prob_correction(gaussian_actions))
        return log_prob

    @override
    def entropy(self) -> torch.Tensor | None:
        if self.output_bijector is not None:
            return None
        return self.sum_action_dim(self.distribution.entropy())

    def get_stds(self, log_stds: torch.Tensor) -> torch.Tensor:
        if self.use_stds_expln:
            below_threshold = torch.exp(log_stds) * (log_stds <= 0)
            safe_log_stds = log_stds * (log_stds > 0) + self.epsilon
            above_threshold = (torch.log1p(safe_log_stds) + 1.0) * (log_stds > 0)
            stds = below_threshold + above_threshold
        else:
            stds = torch.exp(log_stds)

        if self.use_individual_action_stds:
            return stds

        return torch.ones((self.latent_dim, self.action_dim), device=log_stds.device) * stds

    def sample_exploration_noise(self, batch_size: int = 1):
        stds = self.get_stds(self.log_stds)
        self.exploration_noise_dist = torchdist.Normal(loc=torch.zeros_like(stds), scale=stds)
        self.exploration_noise_matrices = self.exploration_noise_dist.rsample(torch.Size((batch_size,)))

    def get_noise(self, latent_pi: torch.Tensor) -> torch.Tensor:
        latent_pi = latent_pi if self.learn_sde_features else latent_pi.detach()

        batch_shape = latent_pi.shape[:-1]

        latent_pi = torch.flatten(latent_pi, end_dim=-2).unsqueeze(dim=1)

        noise = torch.bmm(latent_pi, self.exploration_noise_matrices)
        noise = noise.reshape(batch_shape + (self.action_dim,))

        return noise
