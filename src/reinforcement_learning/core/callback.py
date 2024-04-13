from typing import Generic, TypeVar, Callable, Any

T = TypeVar('T')
CallbackWithStep = Callable[[T, int, dict[str, Any]], None]

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

