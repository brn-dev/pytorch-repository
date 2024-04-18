from typing import Generic, TypeVar, Callable

from src.reinforcement_learning.core.infos import InfoDict

T = TypeVar('T')
CallbackWithStep = Callable[[T, int, InfoDict], None]

class Callback(Generic[T]):

    on_rollout_done: CallbackWithStep
    on_optimization_done: CallbackWithStep

    def __init__(
            self,
            on_rollout_done: CallbackWithStep = lambda rl_algo, step, info: None,
            on_optimization_done: CallbackWithStep = lambda rl_algo, step, info: None,
    ):
        self.on_rollout_done = on_rollout_done
        self.on_optimization_done = on_optimization_done

