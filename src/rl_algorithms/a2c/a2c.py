from typing import Callable, Any

import gymnasium
import torch
from torch import nn, optim

from src.rl_algorithms.rl_utils import compute_returns, normalize


class A2C:
    class RolloutMemory:
        def __init__(self):
            self.action_log_probs: list[torch.Tensor] = []
            self.value_estimates: list[torch.Tensor] = []
            self.rewards: list[float] = []

        def memorize(self, action_log_prob: torch.Tensor, value_estimate: torch.Tensor, reward: float):
            self.action_log_probs.append(action_log_prob)
            self.value_estimates.append(value_estimate)
            self.rewards.append(reward)

        def reset(self):
            del self.action_log_probs[:]
            del self.value_estimates[:]
            del self.rewards[:]

    def __init__(
            self,
            env: gymnasium.Env,
            actor_network: nn.Module,
            actor_network_optimizer: optim.Optimizer,
            critic_network: nn.Module,
            critic_network_optimizer: optim.Optimizer,
            critic_loss: nn.Module,
            # TODO: entropy
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            gamma=0.99,
            normalize_advantage=True,
            on_episode_done: Callable[['A2C', bool, float], None]
                = lambda _self, is_best_episode, best_total_reward: None,
            on_optimization_done: Callable[['A2C', bool, float], None]
                = lambda _self, is_best_episode, best_total_reward: None,
    ):
        self.env = env

        self.actor_network = actor_network
        self.actor_network_optimizer = actor_network_optimizer

        self.critic_network = critic_network
        self.critic_network_optimizer = critic_network_optimizer
        self.critic_loss = critic_loss

        self.select_action = select_action
        self.gamma = gamma
        self.normalize_advantage = normalize_advantage
        self.on_episode_done = on_episode_done
        self.on_optimization_done = on_optimization_done

        self.memory = A2C.RolloutMemory()


    def optimize_using_episode(self):
        returns = compute_returns(self.memory.rewards, gamma=self.gamma, normalize_returns=False)
        action_log_probs = torch.stack(self.memory.action_log_probs)
        value_estimates = torch.stack(self.memory.value_estimates)

        advantages = normalize(returns - value_estimates.detach())

        actor_objective = -(action_log_probs * advantages).mean()
        critic_objective = self.critic_loss(value_estimates, returns)

        self.actor_network_optimizer.zero_grad()
        self.critic_network_optimizer.zero_grad()

        actor_objective.backward()
        critic_objective.backward()

        self.actor_network_optimizer.step()
        self.critic_network_optimizer.step()


    def find_optimal_policy(self, num_episodes: int):
        best_total_reward = 0

        for i_episode in range(num_episodes):
            state, _ = self.env.reset()
            info = {}

            max_timestep = 1000000
            timestep = 0
            for timestep in range(1, max_timestep):  # Don't infinite loop while learning
                action_pred = self.actor_network(torch.tensor(state).float())
                action, action_log_prob = self.select_action(action_pred)

                value_estimate = self.critic_network(state)

                state, reward, done, truncated, info = self.env.step(action)

                self.memory.memorize(action_log_prob, value_estimate, float(reward))

                if done:
                    break

            if timestep == max_timestep - 1:
                info['termination_reason'] = 'timestep_limit_reached'

            episode_total_reward = sum(self.memory.rewards)

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

            self.memory.reset()
