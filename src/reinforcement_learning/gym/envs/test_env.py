from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import ObsType, ActType


class TestEnv(gymnasium.Env):

    def __init__(self, obs_size: int):
        self.counter = 0
        self.obs_size = obs_size

        self.observation_space = gymnasium.spaces.Box(0, 10, (obs_size,))
        self.action_space = gymnasium.spaces.Box(-1, 1, (2,))

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.counter = 0
        return np.array([self.counter] * self.obs_size, dtype=float), {}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.counter += 1
        done = self.counter >= 10

        if False and done:
            reward = 5.0
        else:
            reward = 1.0

        return np.array([self.counter] * self.obs_size, dtype=float), reward, done, False, {}
