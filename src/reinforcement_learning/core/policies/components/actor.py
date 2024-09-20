from typing import Optional

import torch
from torch import nn

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.action_selectors.state_dependent_noise_action_selector import \
    StateDependentNoiseActionSelector
from src.reinforcement_learning.core.policies.components.base_component import BasePolicyComponent
from src.reinforcement_learning.core.policies.components.feature_extractors import FeatureExtractor, IdentityExtractor
from src.reinforcement_learning.core.type_aliases import TensorObs


class Actor(BasePolicyComponent):

    action_selector: ActionSelector
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

    def forward(self, obs: TensorObs) -> ActionSelector:
        obs = self.feature_extractor(obs)
        latent_pi = self.network(obs)
        return self.action_selector.update_latent_features(latent_pi)

    def act_with_log_probs(self, obs: TensorObs) -> tuple[torch.Tensor, torch.Tensor]:
        action_selector: ActionSelector = self(obs)
        actions = action_selector.get_actions()
        actions_log_prob = action_selector.log_prob(actions)
        return actions, actions_log_prob

    def replace_action_selector(self, new_action_selector: ActionSelector, copy_action_net_weights: bool) -> None:
        if copy_action_net_weights:
            new_action_selector.action_net.load_state_dict(self.action_selector.action_net.state_dict())
        self.action_selector = new_action_selector
        self.uses_sde = isinstance(self.action_selector, StateDependentNoiseActionSelector)

    def reset_sde_noise(self, batch_size: int = 1) -> None:
        if self.uses_sde:
            self.action_selector: StateDependentNoiseActionSelector

            self.action_selector.sample_exploration_noise(batch_size)
