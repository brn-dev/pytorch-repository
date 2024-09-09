from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import ObsType, ActType


class TestEnv(gymnasium.Env):

    def __init__(self, obs_size: int, dict_obs: bool, episode_length: int):
        self.counter = 0
        self.obs_size = obs_size
        self.dict_obs = dict_obs
        self.episode_length = episode_length

        if dict_obs:
            self.observation_space = gymnasium.spaces.Dict({
                'a': gymnasium.spaces.Box(0, 10, (obs_size,)),
                'b': gymnasium.spaces.Box(0, 10, (obs_size,)),
            })
        else:
            self.observation_space = gymnasium.spaces.Box(0, 10, (obs_size,))

        self.action_space = gymnasium.spaces.Box(-1, 1, (2,))

    def get_obs(self):
        if self.dict_obs:
            return {
                'a': np.array([self.counter] * self.obs_size, dtype=float),
                'b': np.ones(self.obs_size, dtype=float)
            }
        else:
            return np.array([self.counter] * self.obs_size, dtype=float)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.counter = 0
        return self.get_obs(), {}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.counter += 1
        done = self.counter >= self.episode_length

        if False and done:
            reward = 5.0
        else:
            reward = 1.0

        return self.get_obs(), reward, done, False, {}
