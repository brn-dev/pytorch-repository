from typing import Callable

import gymnasium
from gymnasium import Env
from gymnasium.vector import VectorEnv
from gymnasium.wrappers import AutoResetWrapper


def parallelize_env_async(create_env_fn: Callable[[], Env], num_envs: int) -> VectorEnv:
    return gymnasium.vector.AsyncVectorEnv([lambda: AutoResetWrapper(create_env_fn())] * num_envs)
