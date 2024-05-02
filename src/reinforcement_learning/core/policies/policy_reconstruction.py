from typing import Any

from src.reinforcement_learning.core.policies.base_policy import BasePolicy


def reconstruct_policy(
        serialized_init_policy: str,
        _globals: dict[str, Any]
) -> BasePolicy:
    _locals = {}
    exec(serialized_init_policy, _globals, _locals)
    return _locals['init_policy']()
