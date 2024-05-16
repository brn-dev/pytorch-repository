import torch
from overrides import override
from torch import nn

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy, VALUE_ESTIMATES_KEY

STATE_PREDS_KEY = 'state_preds'


class ActorCriticSTMPolicy(ActorCriticPolicy):

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
        latent_pi, value_estimates, state_preds = self.__predict_latent_pi_values_and_states(obs)
        action_selector = self.action_selector.update_latent_features(latent_pi)
        return action_selector, {VALUE_ESTIMATES_KEY: value_estimates, STATE_PREDS_KEY: state_preds}

    @override
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        return self.__predict_latent_pi_values_and_states(obs)[1]

    def predict_values_and_states(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent_pi, value_estimates, state_preds = self.__predict_latent_pi_values_and_states(obs)
        return value_estimates, state_preds

    def __predict_latent_pi_values_and_states(self, obs: torch.Tensor) \
            -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        latent_pi, value_estimates, state_preds = self(obs_tensor)
        return latent_pi, value_estimates, state_preds
