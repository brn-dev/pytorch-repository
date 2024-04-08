from typing import Callable, Any

import gymnasium
import numpy as np
import torch
from torch import nn, optim

from src.reinforcement_learning.core.episodic_rl_base import EpisodicRLBase, EpisodeDoneCallback

class CombinedNetworkA2C(EpisodicRLBase):
    class RolloutMemory(EpisodicRLBase.RolloutMemory):
        def __init__(self):
            self.action_log_probs: list[torch.Tensor] = []
            self.value_estimates: list[torch.Tensor] = []
            self.rewards: list[float] = []

        def memorize(self, action_log_prob: torch.Tensor, value_estimate: torch.Tensor, reward: float):
            self.action_log_probs.append(action_log_prob)
            self.value_estimates.append(value_estimate)
            self.rewards.append(reward)

    memory: RolloutMemory

    def __init__(
            self,
            env: gymnasium.Env,
            combined_network: nn.Module,
            combined_network_optimizer: optim.Optimizer,
            critic_loss: nn.Module,
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            gamma=0.99,
            critic_objective_weight=0.5,
            on_episode_done: EpisodeDoneCallback['CombinedNetworkA2C'] = lambda _self, info: None,
            on_optimization_done: EpisodeDoneCallback['CombinedNetworkA2C'] = lambda _self, info: None,
    ):
        super().__init__(
            env=env,
            select_action=select_action,
            gamma=gamma,
            on_episode_done=on_episode_done,
            on_optimization_done=on_optimization_done,
        )

        self.combined_network = combined_network
        self.combined_network_optimizer = combined_network_optimizer

        self.critic_loss = critic_loss
        self.critic_objective_weight = critic_objective_weight


    def optimize_using_episode(self):
        returns = self.compute_returns(self.memory.rewards, gamma=self.gamma, normalize_returns=False)
        action_log_probs = torch.stack(self.memory.action_log_probs)
        value_estimates = torch.stack(self.memory.value_estimates)

        advantages = returns - value_estimates.detach()

        actor_objective = -(action_log_probs * advantages).mean()
        critic_objective = self.critic_loss(value_estimates, returns)

        combined_objective = actor_objective + self.critic_objective_weight * critic_objective

        self.combined_network_optimizer.zero_grad()
        combined_objective.backward()
        self.combined_network_optimizer.step()


    def step(self, state: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_pred, value_estimate = self.combined_network(torch.tensor(state).float())
        action, action_log_prob = self.select_action(action_pred)

        state, reward, done, truncated, info = self.env.step(action)
        reward = float(reward)

        self.memory.memorize(action_log_prob, value_estimate, reward)

        return state, reward, done, truncated, info
