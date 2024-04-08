from typing import Callable, Any

import gymnasium
import numpy as np
import torch
from torch import nn, optim

from src.reinforcement_learning.core.episodic_rl_base import EpisodicRLBase, EpisodeDoneCallback

class A2CSTM(EpisodicRLBase):
    class RolloutMemory(EpisodicRLBase.RolloutMemory):
        def __init__(self):
            self.action_log_probs: list[torch.Tensor] = []
            self.value_estimates: list[torch.Tensor] = []
            self.state_preds: list[torch.Tensor] = []
            self.state_targets: list[torch.Tensor] = []
            self.rewards: list[float] = []

        def memorize(
                self,
                action_log_prob: torch.Tensor,
                value_estimate: torch.Tensor,
                state_pred: torch.Tensor,
                state_target: torch.Tensor,
                reward: float
        ):
            self.action_log_probs.append(action_log_prob)
            self.value_estimates.append(value_estimate)
            self.state_preds.append(state_target)
            self.state_targets.append(state_target)
            self.rewards.append(reward)

        def reset(self):
            del self.action_log_probs[:]
            del self.value_estimates[:]
            del self.state_preds[:]
            del self.state_targets[:]
            del self.rewards[:]

    memory: RolloutMemory

    def __init__(
            self,
            env: gymnasium.Env,
            combined_network: nn.Module,
            combined_network_optimizer: optim.Optimizer,
            critic_loss: nn.Module,
            state_transition_loss: nn.Module,
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            gamma=0.99,
            critic_objective_weight=0.5,
            state_transition_objective_weight=0.1,
            on_episode_done: EpisodeDoneCallback['A2CSTM'] = lambda _self, info: None,
            on_optimization_done: EpisodeDoneCallback['A2CSTM'] = lambda _self, info: None,
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

        self.state_transition_loss = state_transition_loss
        self.state_transition_objective_weight = state_transition_objective_weight


    def optimize_using_episode(self):
        returns = self.compute_returns(self.memory.rewards, gamma=self.gamma, normalize_returns=False)
        action_log_probs = torch.stack(self.memory.action_log_probs)
        value_estimates = torch.stack(self.memory.value_estimates)
        state_preds = torch.stack(self.memory.state_preds)
        state_targets = torch.stack(self.memory.state_targets)

        advantages = returns - value_estimates.detach()

        actor_objective = -(action_log_probs * advantages).mean()
        critic_objective = self.critic_loss(value_estimates, returns)
        state_transition_objective = self.state_transition_loss(state_preds, state_targets)

        combined_objective = (actor_objective
                              + self.critic_objective_weight * critic_objective
                              + self.state_transition_objective_weight * state_transition_objective)

        self.combined_network_optimizer.zero_grad()
        combined_objective.backward()
        self.combined_network_optimizer.step()


    def step(self, state: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_pred, value_estimate, state_pred = self.combined_network(torch.tensor(state).float())
        action, action_log_prob = self.select_action(action_pred)

        state, reward, done, truncated, info = self.env.step(action)
        reward = float(reward)

        self.memory.memorize(action_log_prob, value_estimate, state_pred, state, reward)

        return state, reward, done, truncated, info
