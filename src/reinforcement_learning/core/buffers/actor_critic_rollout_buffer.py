from overrides import override

import numpy as np
import torch

from src.reinforcement_learning.core.buffers.basic_rollout_buffer import BasicRolloutBuffer


class ActorCriticRolloutBuffer(BasicRolloutBuffer):
    def __init__(self, buffer_size: int, num_envs: int, obs_shape: tuple[int, ...]):
        super().__init__(buffer_size, num_envs, obs_shape)

        self.value_estimates: list[torch.Tensor] = []

    @override
    def reset(self):
        super().reset()
        del self.value_estimates[:]

    @override
    def add(
            self,
            observations: np.ndarray,
            rewards: np.ndarray,
            episode_starts: np.ndarray,
            action_log_probs: torch.Tensor,
            value_estimates: torch.Tensor = None,
            **extra_predictions: torch.Tensor
    ):
        assert value_estimates is not None
        self.value_estimates.append(value_estimates)
        super().add(
            observations=observations,
            rewards=rewards,
            episode_starts=episode_starts,
            action_log_probs=action_log_probs,
        )
