import torch
from overrides import override
from torch import nn

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.policies.base_policy import BasePolicy

VALUE_ESTIMATES_KEY = 'value_estimates'


class ActorCriticPolicy(BasePolicy):

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
        latent_pi, value_estimates = self.predict_latent_pi_and_values(obs)
        return self.action_selector.update_latent_features(latent_pi), {VALUE_ESTIMATES_KEY: value_estimates}

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.predict_latent_pi_and_values(obs)[1]

    def predict_latent_pi_and_values(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        actions_logits, value_estimates = self(obs_tensor)
        return actions_logits, value_estimates
