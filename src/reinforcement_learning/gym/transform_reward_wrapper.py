from typing import Callable

import gymnasium
import numpy as np
from gymnasium.vector import VectorEnv


class TransformRewardWrapper(gymnasium.core.Wrapper):

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

        if 'raw_rewards' not in infos:
            infos['raw_rewards'] = rewards

        rewards = self.transform_reward(rewards)

        return observations, rewards, terminated, truncated, infos
