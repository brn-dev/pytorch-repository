import copy
from typing import Optional

import torch

from src.console import print_warning
from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.action_selectors.predicted_std_action_selector import PredictedStdActionSelector
from src.reinforcement_learning.core.action_selectors.state_dependent_noise_action_selector import \
    StateDependentNoiseActionSelector
from src.reinforcement_learning.core.buffers.replay.base_replay_buffer import ReplayBufferSamples
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policies.components.actor import Actor
from src.reinforcement_learning.core.policies.components.feature_extractors import FeatureExtractor
from src.reinforcement_learning.core.policies.components.q_critic import QCritic
from src.reinforcement_learning.core.polyak_update import polyak_update
from src.reinforcement_learning.core.type_aliases import TensorObs, detach_obs

"""
        
        Soft Actor-Critic:
        Off-Policy Maximum Entropy Deep Reinforcement
        Learning with a Stochastic Actor
        https://arxiv.org/pdf/1801.01290

"""
class SACPolicy(BasePolicy):

    def __init__(
            self,
            actor: Actor,
            critic: QCritic,
            shared_feature_extractor: Optional[FeatureExtractor] = None
    ):
        super().__init__(actor, shared_feature_extractor)

        self.critic = critic

        self._build_target()

        self._check_action_selector()

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'critic': self.critic.collect_hyper_parameters()
        })

    def _check_action_selector(self):
        if not isinstance(self.actor.action_selector, (PredictedStdActionSelector, StateDependentNoiseActionSelector)):
            print_warning('SAC not being used with PredictedStdAction Selector or gSDE. LogStds should be clamped!')

    def _build_target(self):
        self.target_critic = copy.deepcopy(self.critic)
        self.target_critic.set_trainable(False)

        self.target_shared_feature_extractor = copy.deepcopy(self.shared_feature_extractor)
        self.target_shared_feature_extractor.set_trainable(False)

    def forward(self):
        raise NotImplementedError('forward is not used in SACPolicy')

    def compute_target_values(
            self,
            replay_samples: ReplayBufferSamples,
            entropy_coef: torch.Tensor,
            gamma: float,
    ):
        with torch.no_grad():
            next_observations = replay_samples.next_observations

            next_actions, next_actions_log_prob = self.actor.get_actions_with_log_probs(
                self.shared_feature_extractor(next_observations)
            )

            next_q_values = torch.cat(
                self.target_critic(self.target_shared_feature_extractor(next_observations), next_actions),
                dim=-1
            )
            next_q_values, _ = torch.min(next_q_values, dim=-1, keepdim=True)
            next_q_values = next_q_values - entropy_coef * next_actions_log_prob.reshape(-1, 1)

            target_q_values = replay_samples.rewards + (1 - replay_samples.dones) * gamma * next_q_values

            return target_q_values


    def perform_polyak_update(self, tau: float):
        polyak_update(self.critic.parameters(), self.target_critic.parameters(), tau)
        polyak_update(
            self.shared_feature_extractor.parameters(),
            self.target_shared_feature_extractor.parameters(),
            tau
        )

    def set_train_mode(self, mode: bool) -> None:
        self.actor.set_train_mode(mode)
        self.critic.set_train_mode(mode)
        # Leaving target_critic on train_mode = False

        self.shared_feature_extractor.set_train_mode(mode)
        # Leaving target_shared_feature_extractor on train_mode = False

        self.train_mode = mode
