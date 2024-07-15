import torch
from overrides import override
from torch import nn

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.policies.base_policy import BasePolicy


class ActorPolicy(BasePolicy):

    def __init__(
            self,
            network: nn.Module,
            action_selector: ActionSelector
    ):
        super().__init__(
            network=network,
            action_selector=action_selector
        )

    @override
    def process_obs(self, obs: torch.Tensor) -> tuple[ActionSelector, dict[str, torch.Tensor]]:
        latent_pi = self(obs)
        return self.action_selector.update_latent_features(latent_pi), {}
