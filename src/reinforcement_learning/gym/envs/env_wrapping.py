from gymnasium import Env
from gymnasium.vector import VectorEnv


def wrap_env_using_source(
        env: Env,
        wrap_env_source_code: str,
) -> VectorEnv:
    _globals = {}
    _locals = {}
    exec(wrap_env_source_code, _globals, _locals)
    return _locals['wrap_env'](env)
