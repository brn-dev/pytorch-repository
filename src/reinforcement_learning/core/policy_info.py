from typing import TypedDict


class PolicyInfo(TypedDict):
    score: float
    steps_trained: int
    init_policy_source_code: str
    wrap_env_source_code: str

