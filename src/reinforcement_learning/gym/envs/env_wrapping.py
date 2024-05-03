from typing import Any

import gymnasium
from gymnasium import Env
from gymnasium.vector import VectorEnv

from src.reinforcement_learning.gym.envs.parallelize_env import parallelize_env_async


def wrap_env_using_source(
        env: Env,
        wrap_env_source_code: str,
        _globals: dict[str, Any],
) -> VectorEnv:
    _locals = {}
    exec(wrap_env_source_code, _globals, _locals)
    return _locals['wrap_env'](env)
