from typing import TypedDict

from src.reinforcement_learning.core.policy_construction import PolicyInitializationInfo


class MitosisPolicyInfo(TypedDict):
    policy_id: str
    parent_policy_id: str | None

    score: float

    steps_trained: int
    env_steps_trained: int
    optimizations_done: int

    initialization_info: PolicyInitializationInfo


