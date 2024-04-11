import numpy as np
import torch


class BasicRolloutBuffer:
    def __init__(self, buffer_size: int, num_envs: int, obs_shape: tuple[int, ...]):
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.obs_shape = obs_shape

        self.states = np.zeros((buffer_size,) + obs_shape, dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_envs), dtype=np.float32)
        self.episode_starts = np.zeros((buffer_size, num_envs)).astype(bool)
        self.action_log_probs: list[torch.Tensor] = []
        self.value_estimates: list[torch.Tensor] = []

        self.pos = 0

    @property
    def full(self):
        return self.pos == self.buffer_size

    def reset(self):
        self.states = np.zeros((self.buffer_size, self.num_envs) + self.obs_shape)
        self.rewards = np.zeros((self.buffer_size, self.num_envs))
        self.episode_starts = np.zeros((self.buffer_size, self.num_envs)).astype(bool)

        del self.action_log_probs[:]
        del self.value_estimates[:]

        self.pos = 0

    def add(
            self,
            states: np.ndarray,
            rewards: np.ndarray,
            episode_starts: np.ndarray,
            action_log_probs: torch.Tensor,
            value_estimates: torch.Tensor,
    ):
        assert not self.full

        self.states[self.pos] = states
        self.rewards[self.pos] = rewards
        self.episode_starts[self.pos] = episode_starts

        self.action_log_probs.append(action_log_probs)
        self.value_estimates.append(value_estimates)

        self.pos += 1
