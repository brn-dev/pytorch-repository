import gymnasium
import numpy as np
import torch

from src.reinforcement_learning.core.generalized_advantage_estimate import compute_episode_returns
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.gym.singleton_vector_env import as_vec_env
from src.reinforcement_learning.gym.wrappers.reward_wrapper import RewardWrapper
from src.torch_device import TorchDevice


def evaluate_policy(
        env: gymnasium.Env,
        policy: BasePolicy,
        num_steps: int,
        gamma: float = 1.0,
        gae_lambda: float = 1.0,
        deterministic_actions: bool = False,
        remove_unfinished_episodes: bool = True,
        torch_device: TorchDevice = 'cpu'
) -> np.ndarray:
    policy.eval()
    policy = policy.to(torch_device)

    env, num_envs = as_vec_env(env)

    obs, _ = env.reset()
    episode_starts = np.zeros((num_steps, num_envs), dtype=bool)
    last_episode_starts = np.zeros((num_envs,), dtype=bool)

    rollout_rewards = np.zeros((num_steps, num_envs))

    with torch.no_grad():
        for step in range(num_steps):
            action_selector, extra_predictions = policy.process_obs(torch.tensor(obs, device=torch_device))
            actions = action_selector.get_actions(deterministic=deterministic_actions)
            obs, rewards, terminated, truncated, info = env.step(actions.detach().cpu().numpy())

            if RewardWrapper.RAW_REWARDS_KEY in info:
                rollout_rewards[step] = info[RewardWrapper.RAW_REWARDS_KEY]
            else:
                rollout_rewards[step] = rewards

            last_episode_starts = np.logical_or(terminated, truncated)
            episode_starts[step] = last_episode_starts

    episode_returns = compute_episode_returns(
        rewards=rollout_rewards,
        episode_starts=episode_starts,
        last_episode_starts=last_episode_starts,
        gamma=gamma,
        gae_lambda=gae_lambda,
        normalize_rewards=None,
        remove_unfinished_episodes=remove_unfinished_episodes,
    )

    print(episode_starts.astype(int).sum(axis=0).mean())

    return episode_returns
