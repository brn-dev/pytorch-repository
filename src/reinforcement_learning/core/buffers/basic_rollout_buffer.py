import numpy as np
import torch


class BasicRolloutBuffer:
    def __init__(self, buffer_size: int, num_envs: int, obs_shape: tuple[int, ...], detach_actions: bool = True):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_shape = obs_shape

        self.observations = np.zeros((buffer_size,) + obs_shape, dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.episode_starts = np.zeros((buffer_size, num_envs)).astype(bool)

        self.actions: list[torch.Tensor] = []
        self.action_log_probs: list[torch.Tensor] = []

        self.detach_actions = detach_actions

        self.pos = 0

    @property
    def full(self):
        return self.pos == self.buffer_size

    def reset(self):
        self.observations = np.zeros((self.buffer_size,) + self.obs_shape)
        self.rewards = np.zeros((self.buffer_size, self.num_envs))
        self.episode_starts = np.zeros((self.buffer_size, self.num_envs)).astype(bool)

        del self.actions[:]
        del self.action_log_probs[:]

        self.pos = 0

    def add(
            self,
            observations: np.ndarray,
            rewards: np.ndarray,
            episode_starts: np.ndarray,
            actions: torch.Tensor,
            action_log_probs: torch.Tensor,
            **extra_predictions: torch.Tensor
    ):
        assert not self.full
        assert not extra_predictions

        self.observations[self.pos] = observations
        self.rewards[self.pos] = rewards
        self.episode_starts[self.pos] = episode_starts

        self.actions.append(actions.detach() if self.detach_actions else actions)
        self.action_log_probs.append(action_log_probs)

        self.pos += 1
