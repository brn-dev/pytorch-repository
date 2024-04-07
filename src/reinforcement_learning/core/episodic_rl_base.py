import abc
from typing import Callable, Any, TypeVar

import gymnasium
import numpy as np
import torch

from src.reinforcement_learning.core.rl_base import RLBase


class EpisodicRLBase(RLBase, abc.ABC):
    class RolloutMemory(abc.ABC):

        action_log_probs: list[torch.Tensor]
        rewards: list[float]

        @abc.abstractmethod
        def memorize(self, *args):
            raise NotImplemented

        @abc.abstractmethod
        def reset(self):
            raise NotImplemented


    def __init__(
            self,
            env: gymnasium.Env,
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            gamma,
            on_episode_done: 'EpisodeDoneCallback',
            on_optimization_done: 'EpisodeDoneCallback',
    ):
        self.env = env
        self.select_action = select_action
        self.gamma = gamma
        self.on_episode_done = on_episode_done
        self.on_optimization_done = on_optimization_done
        self.memory: RolloutMemoryDerived = self.RolloutMemory()

    @abc.abstractmethod
    def optimize_using_episode(self) -> None:
        raise NotImplemented

    @abc.abstractmethod
    def step(self, state: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        raise NotImplemented

    def find_optimal_policy(self, num_episodes: int, *args, **kwargs):
        best_total_reward = 0

        for i_episode in range(num_episodes):
            state, _ = self.env.reset()
            info = {}

            max_timestep = 1000000
            timestep = 0
            for timestep in range(1, max_timestep + 1):  # Don't infinite loop while learning
                state, reward, done, truncated, info = self.step(state)

                if done:
                    break

            if timestep == max_timestep:
                info['termination_reason'] = 'timestep_limit_reached'

            episode_total_reward = sum(self.memory.rewards)

            is_best_episode = False
            if episode_total_reward >= best_total_reward:
                best_total_reward = episode_total_reward
                is_best_episode = True

            self.on_episode_done(
                self,
                i_episode,
                is_best_episode,
                episode_total_reward,
                timestep
            )

            self.optimize_using_episode()

            self.on_optimization_done(
                self,
                i_episode,
                is_best_episode,
                episode_total_reward,
                timestep
            )

            self.memory.reset()


EpisodicRLBaseDerived = TypeVar('EpisodicRLBaseDerived', bound=EpisodicRLBase)
EpisodeDoneCallback = Callable[[EpisodicRLBaseDerived, int, bool, float, int], None]
RolloutMemoryDerived = TypeVar('RolloutMemoryDerived', bound=EpisodicRLBase.RolloutMemory)
