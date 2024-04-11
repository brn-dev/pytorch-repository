from typing import Callable, Any

import gymnasium
import numpy as np
import torch
from torch import nn, optim

from src.reinforcement_learning.core.episodic_rl_base import EpisodicRLBase, RolloutDoneCallback


class A2C(EpisodicRLBase):

    def __init__(
            self,
            env: gymnasium.Env,
            actor_network: nn.Module,
            actor_network_optimizer: optim.Optimizer,
            critic_network: nn.Module,
            critic_network_optimizer: optim.Optimizer,
            critic_loss: nn.Module,
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            gamma=0.99,
            on_rollout_done: RolloutDoneCallback['A2C'] = lambda _self, info: None,
            on_optimization_done: RolloutDoneCallback['A2C'] = lambda _self, info: None,
    ):
        super().__init__(
            env=env,
            select_action=select_action,
            gamma=gamma,
            on_rollout_done=on_rollout_done,
            on_optimization_done=on_optimization_done,
        )
        self.actor_network = actor_network
        self.actor_network_optimizer = actor_network_optimizer

        self.critic_network = critic_network
        self.critic_network_optimizer = critic_network_optimizer
        self.critic_loss = critic_loss

    def optimize(self, info: dict[str, Any]):
        returns = self.compute_returns(self.memory.rewards, gamma=self.gamma, normalize_returns=False)
        action_log_probs = torch.stack(self.memory.action_log_probs)
        value_estimates = torch.stack(self.memory.value_estimates)

        advantages = returns - value_estimates.detach()

        if action_log_probs.dim() == 2:
            advantages = advantages.unsqueeze(1)

        actor_objective = -(action_log_probs * advantages).mean()
        critic_objective = self.critic_loss(value_estimates, returns)

        self.actor_network_optimizer.zero_grad()
        self.critic_network_optimizer.zero_grad()

        actor_objective.backward()
        critic_objective.backward()

        self.actor_network_optimizer.step()
        self.critic_network_optimizer.step()

        info['returns'] = returns
        info['advantages'] = advantages

    def rollout_step(self, state: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        state = torch.tensor(state)

        action_pred = self.actor_network(state.float())
        action, action_log_prob = self.select_action(action_pred)

        value_estimate = self.critic_network(state).squeeze()

        state, reward, done, truncated, info = self.env.rollout_step(action)
        reward = float(reward)

        self.memory.memorize(action_log_prob, value_estimate, reward)

        return state, reward, done, truncated, info
