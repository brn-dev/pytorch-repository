from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import ObsType, ActType


class TestEnv(gymnasium.Env):

    def __init__(self, obs_size: int, dict_obs: bool = False, action_in_obs: bool = False, episode_length: int = 10):
        self.counter = 0

        self.action_size = 2
        self.obs_size = obs_size

        self.dict_obs = dict_obs
        self.action_in_obs = action_in_obs

        self.episode_length = episode_length

        _obs_size = self.obs_size + (self.action_size if self.action_in_obs else 0)
        if dict_obs:
            self.observation_space = gymnasium.spaces.Dict({
                'a': gymnasium.spaces.Box(0, 10, (_obs_size,)),
                'b': gymnasium.spaces.Box(0, 10, (_obs_size,)),
            })
        else:
            self.observation_space = gymnasium.spaces.Box(0, 10, (_obs_size,))

        self.action_space = gymnasium.spaces.Box(-1, 1, (self.action_size,))

    def get_obs(self, action: np.ndarray):
        obs = np.array([self.counter] * self.obs_size, dtype=float)

        if self.action_in_obs:
            obs = np.concatenate((obs, action))

        if self.dict_obs:
            return {
                'a': obs,
                'b': np.ones(self.obs_size, dtype=float)
            }
        else:
            return obs

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.counter = 0
        return self.get_obs(np.zeros(self.action_size, dtype=float)), {}

    def step(
            self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.counter += 1
        done = self.counter >= self.episode_length

        if True and done:
            reward = 5.0
        else:
            reward = 1.0

        return self.get_obs(action), reward, done, done, {}
