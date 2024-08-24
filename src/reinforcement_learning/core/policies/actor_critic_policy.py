import torch
from torch import nn

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.policies.base_policy import BasePolicy, ObsPreprocessing, Obs
from src.reinforcement_learning.core.policies.components.actor import Actor
from src.reinforcement_learning.core.policies.components.v_critic import VCritic

VALUE_ESTIMATES_KEY = 'value_estimates'


class ActorCriticPolicy(BasePolicy):

    def __init__(
            self,
            actor: Actor,
            critic: VCritic | nn.Module,  # A state value critic is essentially just a module
            obs_preprocessing: ObsPreprocessing = nn.Identity()
    ):
        super().__init__(
            actor=actor,
            obs_preprocessing=obs_preprocessing,
        )
        self.critic = critic

    def forward(self, obs: Obs) -> tuple[ActionSelector, torch.Tensor]:
        obs = self.obs_preprocessing(obs)
        return self.actor(obs), self.critic(obs)

    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        obs = self.obs_preprocessing(obs)
        return self.critic(obs)
