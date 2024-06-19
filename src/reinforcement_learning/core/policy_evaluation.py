from typing import Any

import gymnasium
import numpy as np
import torch
from gymnasium.wrappers import AutoResetWrapper, RecordVideo

from src.datetime import get_current_timestamp
from src.reinforcement_learning.core.generalized_advantage_estimate import compute_episode_returns
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policy_construction import WrapEnvFunction
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
    policy.to(torch_device)

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


def record_policy(
        env: gymnasium.Env,
        policy: BasePolicy,
        video_folder: str,
        deterministic_actions: bool,
        num_steps: int,
        wrap_env: WrapEnvFunction = lambda env, hyper_parameter: env,
        wrap_env_hyper_parameters: dict[str, Any] = None,
        torch_device: TorchDevice = 'cpu',
):
    try:
        policy.eval()
        policy.to(torch_device)

        if 'render_fps' not in env.metadata:
            env.metadata['render_fps'] = 30
        env = AutoResetWrapper(RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep_nr: True))
        env = wrap_env(as_vec_env(env)[0], wrap_env_hyper_parameters or {})

        policy.reset_sde_noise(1)

        with torch.no_grad():
            obs, info = env.reset()
            for step in range(num_steps):
                actions_dist, _ = policy.process_obs(torch.tensor(obs, device=torch_device))
                actions = actions_dist.get_actions(deterministic=deterministic_actions).detach().cpu().numpy()
                obs, reward, terminated, truncated, info = env.step(actions)

    except KeyboardInterrupt:
        print('keyboard interrupt')
    finally:
        print('closing record env')
        env.close()
        print('record env closed')
