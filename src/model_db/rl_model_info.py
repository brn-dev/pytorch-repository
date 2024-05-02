from typing import TypedDict


class RLModelInfo(TypedDict):
    score: float
    steps_trained: int
    wrap_env_function: str
