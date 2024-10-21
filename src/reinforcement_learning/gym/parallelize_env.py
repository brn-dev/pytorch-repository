from typing import Callable

from gymnasium import Env
from gymnasium.vector import VectorEnv

from src.reinforcement_learning.gym.vector.async_vector_env_v0291 import AsyncVectorEnvV0291
from src.reinforcement_learning.gym.vector.sync_vector_env_v0291 import SyncVectorEnvV0291

"""
    Using VectorEnvs from Gymnasium v0.29.1 to preserve the pre-1.0 auto-reset behavior 
"""


def parallelize_env_async(create_env_fn: Callable[[], Env], num_envs: int) -> VectorEnv:
    return AsyncVectorEnvV0291([create_env_fn] * num_envs)


def parallelize_env_sync(create_env_fn: Callable[[], Env], num_envs: int) -> VectorEnv:
    return SyncVectorEnvV0291([create_env_fn] * num_envs)
