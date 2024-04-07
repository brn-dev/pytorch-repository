from typing import Callable, Any

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim

from src.reinforcement_learning.core.episodic_rl_base import EpisodicRLBase, EpisodeDoneCallback


class ReinforceSTM(EpisodicRLBase):

    class RolloutMemory(EpisodicRLBase.RolloutMemory):

        def __init__(self):
            self.action_log_probs: list[torch.Tensor] = []
            self.state_preds: list[torch.Tensor] = []
            self.state_targets: list[torch.Tensor] = []
            self.rewards: list[float] = []

        def memorize(
                self,
                action_log_prob: torch.Tensor,
                state_pred: torch.Tensor,
                state_target: torch.Tensor,
                reward: float
        ):
            self.action_log_probs.append(action_log_prob)
            self.state_preds.append(state_target)
            self.state_targets.append(state_target)
            self.rewards.append(reward)

        def reset(self):
            del self.action_log_probs[:]
            del self.state_preds[:]
            del self.state_targets[:]
            del self.rewards[:]

    memory: RolloutMemory

    def __init__(
            self,
            env: gymnasium.Env,
            policy_network: nn.Module,
            policy_network_optimizer: optim.Optimizer,
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            gamma=0.99,
            normalize_returns=True,
            reinforce_objective_weight=10.0,
            state_transition_objective_weight=1.0,
            on_episode_done: EpisodeDoneCallback['ReinforceSTM']
                = lambda _self, i_episode, is_best_episode, best_total_reward, end_timestep: None,
            on_optimization_done: EpisodeDoneCallback['ReinforceSTM']
                = lambda _self, i_episode, is_best_episode, best_total_reward, end_timestep: None,
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
        self.normalize_returns = normalize_returns
        self.reinforce_objective_weight = reinforce_objective_weight
        self.state_transition_objective_weight = state_transition_objective_weight

    def optimize_using_episode(self):
        returns = self.compute_returns(self.memory.rewards, gamma=self.gamma, normalize_returns=self.normalize_returns)
        action_log_probs = torch.stack(self.memory.action_log_probs)

        reinforce_objective = -(action_log_probs * returns).mean()
        state_transition_objective = F.mse_loss(
            torch.stack(self.memory.state_preds),
            torch.stack(self.memory.state_targets)
        )
        combined_objective = (self.reinforce_objective_weight * reinforce_objective +
                              self.state_transition_objective_weight * state_transition_objective)

        self.policy_network_optimizer.zero_grad()
        combined_objective.backward()
        self.policy_network_optimizer.step()

        return returns, reinforce_objective, state_transition_objective


    def step(self, state: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action_pred, state_pred = self.policy_network(torch.tensor(state))
        action, action_log_probs = self.select_action(action_pred)

        state, reward, done, truncated, info = self.env.step(action)
        reward = float(reward)

        self.memory.memorize(action_log_probs, state_pred, torch.tensor(state), reward)

        return state, reward, done, truncated, info
