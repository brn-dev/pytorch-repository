from typing import Callable

import gymnasium
from gymnasium import Env
from gymnasium.vector import VectorEnv


def parallelize_env_async(create_env_fn: Callable[[], Env], num_envs: int) -> VectorEnv:
    return gymnasium.vector.AsyncVectorEnv([create_env_fn] * num_envs)

def parallelize_env_sync(create_env_fn: Callable[[], Env], num_envs: int) -> VectorEnv:
    return gymnasium.vector.SyncVectorEnv([create_env_fn] * num_envs)
