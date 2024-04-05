from typing import Callable, Any

import gymnasium
import numpy as np
import torch
from torch import nn, optim

from src.reinforcement_learning.core.rl_base import RLBase
from src.reinforcement_learning.rl_utils import compute_returns


class Reinforce(RLBase):

    class RolloutMemory(RLBase.RolloutMemory):

        def __init__(self):
            self.action_log_probs = []
            self.rewards = []

        def memorize(self, action_log_prob: torch.Tensor, reward: float):
            self.action_log_probs.append(action_log_prob)
            self.rewards.append(reward)

        def reset(self):
            del self.action_log_probs[:]
            del self.rewards[:]

    memory: RolloutMemory

    def __init__(
            self,
            env: gymnasium.Env,
            policy_network: nn.Module,
            policy_network_optimizer: optim.Optimizer,
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            gamma=0.99,
            on_episode_done: Callable[['Reinforce', int, bool, float], None]
                = lambda _self, i_episode, is_best_episode, best_total_reward: None,
            on_optimization_done: Callable[['Reinforce', int, bool, float], None]
                = lambda _self, i_episode, is_best_episode, best_total_reward: None,
    ):
        super().__init__(
            env=env,
            select_action=select_action,
            gamma=gamma,
            on_episode_done=on_episode_done,
            on_optimization_done=on_optimization_done,
        )
        self.policy_network = policy_network
        self.policy_network_optimizer = policy_network_optimizer

    def optimize_using_episode(self):
        returns = compute_returns(self.memory.rewards, gamma=self.gamma, normalize_returns=True)
        action_log_probs = torch.stack(self.memory.action_log_probs)

        reinforce_objective = -(action_log_probs * returns).mean()

        self.policy_network_optimizer.zero_grad()
        reinforce_objective.backward()
        self.policy_network_optimizer.step()

    def step(self, state: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_pred = self.policy_network(torch.tensor(state).float())
        action, action_log_probs = self.select_action(action_pred)
        state, reward, done, truncated, info = self.env.step(action)
        reward = float(reward)

        self.memory.memorize(action_log_probs, reward)

        return state, reward, done, truncated, info


