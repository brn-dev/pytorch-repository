from typing import Optional

import torch
from torch import nn

from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.action_selectors.state_dependent_noise_action_selector import \
    StateDependentNoiseActionSelector
from src.reinforcement_learning.core.policies.components.base_component import BasePolicyComponent
from src.reinforcement_learning.core.policies.components.feature_extractors import FeatureExtractor, IdentityExtractor
from src.reinforcement_learning.core.type_aliases import TensorObs


class Actor(BasePolicyComponent):

    _action_selector: ActionSelector
    uses_sde: bool

    def __init__(
            self,
            network: nn.Module,
            action_selector: ActionSelector,
            feature_extractor: Optional[FeatureExtractor] = None
    ):
        super().__init__(feature_extractor or IdentityExtractor())
        self.network = network
        self.replace_action_selector(action_selector, copy_action_net_weights=False)

    @property
    def action_selector(self):
        return self._action_selector

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'network': self.get_hps_or_repr(self.network),
            'action_selector': self.get_hps_or_repr(self._action_selector),
        })

    def forward(self, obs: TensorObs) -> ActionSelector:
        obs = self.feature_extractor(obs)
        latent_pi = self.network(obs)
        return self._action_selector.update_latent_features(latent_pi)

    def get_actions_with_log_probs(
            self,
            obs: TensorObs,
            deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs = self.feature_extractor(obs)
        latent_pi = self.network(obs)
        return self._action_selector.get_actions_with_log_probs(latent_pi, deterministic=deterministic)

    def replace_action_selector(self, new_action_selector: ActionSelector, copy_action_net_weights: bool) -> None:
        if copy_action_net_weights:
            new_action_selector.action_net.load_state_dict(self._action_selector.action_net.state_dict())
        self._action_selector = new_action_selector
        self.uses_sde = isinstance(self._action_selector, StateDependentNoiseActionSelector)

    def reset_sde_noise(self, batch_size: int = 1) -> None:
        if self.uses_sde:
            self._action_selector: StateDependentNoiseActionSelector

            self._action_selector.sample_exploration_noise(batch_size)
