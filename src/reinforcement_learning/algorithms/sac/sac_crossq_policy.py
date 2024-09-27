from typing import Optional

import torch

from src.console import print_warning
from src.normalization import normalization_classes
from src.reinforcement_learning.algorithms.sac.sac_policy import SACPolicy
from src.reinforcement_learning.core.buffers.replay.base_replay_buffer import ReplayBufferSamples
from src.reinforcement_learning.core.policies.components.actor import Actor
from src.reinforcement_learning.core.policies.components.feature_extractors import FeatureExtractor
from src.reinforcement_learning.core.policies.components.q_critic import QCritic
from src.reinforcement_learning.core.type_aliases import TensorObs, detach_obs

"""

        CrossQ: Batch Normalization in Deep Reinforcement Learning for Greater Sample Efficiency and Simplicity
        https://arxiv.org/pdf/1902.05605

"""
class SACCrossQPolicy(SACPolicy):

    target_critic: None
    target_shared_feature_extractor: None

    def __init__(
            self,
            actor: Actor,
            critic: QCritic,
            shared_feature_extractor: Optional[FeatureExtractor] = None
    ):
        super().__init__(actor, critic, shared_feature_extractor)

        critic_has_normalization = any(isinstance(module, normalization_classes) for module in self.critic.modules())

        if not critic_has_normalization:
            print_warning('A CrossQ critic should include normalization!')

    def _build_target(self):
        # target_critic is not used by CrossQ
        pass

    # # Translation of jax to pytorch code from paper (very slow for some reason)
    # def compute_current_and_target_values(
    #         self,
    #         observation_features: TensorObs,
    #         replay_samples: ReplayBufferSamples,
    #         entropy_coef: torch.Tensor,
    #         gamma: float,
    # ):
    #     with torch.no_grad():
    #         next_observation_features = self.shared_feature_extractor(replay_samples.next_observations)
    #
    #         next_actions, next_actions_log_prob = self.actor.act_with_log_probs(
    #             self.shared_feature_extractor(next_observation_features)
    #         )
    #
    #     all_q_values = self.critic(
    #         torch.cat((detach_obs(observation_features), next_observation_features)),
    #         torch.cat((replay_samples.actions, next_actions))
    #     )
    #     all_q_values = torch.cat(all_q_values, dim=-1)
    #     current_q_values, next_q_values = torch.chunk(all_q_values, 2)
    #
    #     next_q_values = torch.min(next_q_values, dim=-1, keepdim=True)[0].detach()
    #     # TODO: subtract entropy term? Not originally in the paper but used in normal SAC calculation
    #     next_q_values = next_q_values - entropy_coef * next_actions_log_prob.reshape(-1, 1)
    #
    #     target_q_values = replay_samples.rewards + (1 - replay_samples.dones) * gamma * next_q_values
    #
    #     return current_q_values, target_q_values
    def compute_target_values(
            self,
            replay_samples: ReplayBufferSamples,
            entropy_coef: torch.Tensor,
            gamma: float,
    ):
        # critic loss should not influence shared feature extractor
        with torch.no_grad():
            next_observation_features = self.shared_feature_extractor(replay_samples.next_observations)

            next_actions, next_actions_log_prob = self.actor.get_actions_with_log_probs(next_observation_features)

            next_q_values = torch.cat(
                self.critic(next_observation_features, next_actions),
                dim=-1
            )
            next_q_values, _ = torch.min(next_q_values, dim=-1, keepdim=True)
            # TODO: subtract entropy term? Not originally in the paper but used in normal SAC calculation
            next_q_values = next_q_values - entropy_coef * next_actions_log_prob.reshape(-1, 1)

            target_q_values = replay_samples.rewards + (1 - replay_samples.dones) * gamma * next_q_values


            return target_q_values

    def perform_polyak_update(self, tau: float):
        # target_critic is not used by CrossQ and does not need to be updated
        pass
