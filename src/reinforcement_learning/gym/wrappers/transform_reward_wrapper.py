from typing import Callable

import gymnasium
import numpy as np
from gymnasium.vector import VectorEnv

from src.reinforcement_learning.gym.wrappers.reward_wrapper import RewardWrapper

class TransformRewardWrapper(RewardWrapper):

    env: VectorEnv

    def __init__(
            self,
            env: VectorEnv,
            transform_reward: Callable[[np.ndarray], np.ndarray]
    ):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, action):
        observations, rewards, terminated, truncated, infos = self.env.step(action)

        if self.RAW_REWARDS_KEY not in infos:
            infos[self.RAW_REWARDS_KEY] = rewards

        rewards = self.transform_reward(rewards)

        return observations, rewards, terminated, truncated, infos

