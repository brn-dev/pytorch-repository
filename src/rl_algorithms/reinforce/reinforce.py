from typing import Callable, Any

import gymnasium
import torch
from torch import nn, optim

from src.rl_algorithms.rl_utils import compute_returns


class Reinforce:
    episode_action_log_probs: list[torch.Tensor] = []
    episode_rewards: list[float] = []

    def __init__(
            self,
            env: gymnasium.Env,
            policy_network: nn.Module,
            policy_network_optimizer: optim.Optimizer,
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            gamma=0.99,
            on_episode_done: Callable[['Reinforce', bool, float], None]
                = lambda _self, is_best_episode, best_total_reward: None,
            on_optimization_done: Callable[['Reinforce', bool, float], None]
                = lambda _self, is_best_episode, best_total_reward: None,
    ):
        self.env = env
        self.policy_network = policy_network
        self.policy_network_optimizer = policy_network_optimizer
        self.select_action = select_action
        self.gamma = gamma
        self.on_episode_done = on_episode_done
        self.on_optimization_done = on_optimization_done

    def optimize_using_episode(self):
        returns = compute_returns(self.episode_rewards, gamma=self.gamma, normalize_returns=True)
        action_log_probs = torch.stack(self.episode_action_log_probs)

        reinforce_objective = -(action_log_probs * returns).mean()

        self.policy_network_optimizer.zero_grad()
        reinforce_objective.backward()
        self.policy_network_optimizer.step()

    def find_optimal_policy(self, num_episodes: int):
        best_total_reward = 0

        for i_episode in range(num_episodes):
            state, _ = self.env.reset()
            info = {}

            max_timestep = 1000000
            timestep = 0
            for timestep in range(1, max_timestep):  # Don't infinite loop while learning
                action_pred = self.policy_network(torch.tensor(state).float())
                action, action_log_probs = self.select_action(action_pred)
                state, reward, done, truncated, info = self.env.step(action)

                self.episode_action_log_probs.append(action_log_probs)
                self.episode_rewards.append(float(reward))

                if done:
                    break

            if timestep == max_timestep - 1:
                info['termination_reason'] = 'timestep_limit_reached'

            episode_total_reward = sum(self.episode_rewards)

            is_best_episode = False
            if episode_total_reward >= best_total_reward:
                best_total_reward = episode_total_reward
                is_best_episode = True

            self.on_episode_done(
                self,
                is_best_episode,
                best_total_reward
            )

            self.optimize_using_episode()

            self.on_optimization_done(
                self,
                is_best_episode,
                best_total_reward
            )

            del self.episode_rewards[:]
            del self.episode_action_log_probs[:]
