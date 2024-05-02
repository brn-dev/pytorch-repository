from typing import Any

import gymnasium
from gymnasium.vector import VectorEnv

from src.reinforcement_learning.gym.envs.parallelize_env import parallelize_env_async


def reconstruct_async_vector_env(
        env_name: str,
        env_render_mode: str | None,
        env_kwargs: dict[str, Any],
        num_envs: int,
        serialized_wrap_env: str,
        _globals: dict[str, Any],
) -> VectorEnv:
    vec_env = parallelize_env_async(
        lambda: gymnasium.make(env_name, render_mode=env_render_mode, **env_kwargs),
        num_envs
    )

    _locals = {}
    exec(serialized_wrap_env, _globals, _locals)

    return _locals['wrap_env'](vec_env)
