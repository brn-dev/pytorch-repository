from typing import TypedDict


class PolicyInfo(TypedDict):
    policy_id: str
    parent_policy_id: str | None

    score: float
    steps_trained: int
    init_policy_source_code: str
    wrap_env_source_code: str

