from overrides import override

import numpy as np
import torch

from src.reinforcement_learning.core.buffers.actor_critic_rollout_buffer import ActorCriticRolloutBuffer


class ActorCriticSTMRolloutBuffer(ActorCriticRolloutBuffer):
    def __init__(self, buffer_size: int, num_envs: int, obs_shape: tuple[int, ...]):
        super().__init__(buffer_size, num_envs, obs_shape)

        self.state_preds: list[torch.Tensor] = []

    @override
    def reset(self):
        super().reset()
        del self.state_preds[:]

    @override
    def add(
            self,
            observations: np.ndarray,
            rewards: np.ndarray,
            episode_starts: np.ndarray,
            action_log_probs: torch.Tensor,
            value_estimates: torch.Tensor = None,
            state_preds: torch.Tensor = None,
            **extra_predictions: torch.Tensor
    ):
        assert state_preds is not None
        self.state_preds.append(state_preds)
        super().add(
            observations=observations,
            rewards=rewards,
            episode_starts=episode_starts,
            action_log_probs=action_log_probs,
            value_estimates=value_estimates,
            **extra_predictions
        )
