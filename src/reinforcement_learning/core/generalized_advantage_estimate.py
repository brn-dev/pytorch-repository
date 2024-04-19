
import numpy as np

from src.reinforcement_learning.core.normalization import normalize_np_array, NormalizationType

# Adapted from
# https://github.com/DLR-RM/stable-baselines3/blob/5623d98f9d6bcfd2ab450e850c3f7b090aef5642/stable_baselines3/common/buffers.py#L402
def compute_gae_and_returns(
        value_estimates: np.ndarray,
        rewards: np.ndarray,
        episode_starts: np.ndarray,
        last_values: np.ndarray,
        last_dones: np.ndarray,
        gamma: float,
        gae_lambda: float,
        normalize_rewards: NormalizationType | None,
        normalize_advantages: NormalizationType | None,
) -> tuple[np.ndarray, np.ndarray]:
    sequence_length = len(value_estimates)

    if normalize_rewards is not None:
        rewards = normalize_np_array(rewards, normalization_type=normalize_rewards)

    advantages = np.zeros_like(rewards)

    gae = 0
    for step in reversed(range(sequence_length)):
        if step == sequence_length - 1:
            next_non_terminal = 1.0 - last_dones
            next_values = last_values
        else:
            next_non_terminal = 1.0 - episode_starts[step + 1]
            next_values = value_estimates[step + 1]
        delta = rewards[step] + gamma * next_values * next_non_terminal - value_estimates[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae

        advantages[step] = gae

    returns = advantages + value_estimates

    if normalize_advantages is not None:
        advantages = normalize_np_array(advantages, normalization_type=normalize_advantages)

    return advantages, returns
