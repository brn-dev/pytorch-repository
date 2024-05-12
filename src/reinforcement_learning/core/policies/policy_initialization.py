from src.reinforcement_learning.core.policies.base_policy import BasePolicy


def init_policy_using_source(
        init_policy_source_code: str,
) -> BasePolicy:
    _globals = {}
    _locals = {}
    exec(init_policy_source_code, _globals, _locals)
    return _locals['init_policy']()
