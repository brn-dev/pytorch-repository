from typing import Generic, TypeVar, Callable, Iterable, Any

from src.reinforcement_learning.algorithms.base.base_algorithm import BaseAlgorithm
from src.reinforcement_learning.core.infos import InfoDict
from src.schedulers import Scheduler

Algo = TypeVar('Algo', bound=BaseAlgorithm)
SchedulerDict = dict[str, Scheduler]
CallbackWithStep = Callable[[Algo, int, InfoDict, dict[str, Any]], None]

class Callback(Generic[Algo]):

    def __init__(
            self,
            on_rollout_done: CallbackWithStep = lambda optim_algo, step, info, scheduler_values: None,
            rollout_schedulers: SchedulerDict = None,
            on_optimization_done: CallbackWithStep = lambda optim_algo, step, info, scheduler_values: None,
            optimization_schedulers: SchedulerDict = None,
    ):
        self._on_rollout_done = on_rollout_done
        self.rollout_schedulers: SchedulerDict = rollout_schedulers or {}

        self._on_optimization_done = on_optimization_done
        self.optimization_schedulers: SchedulerDict = optimization_schedulers or {}


    def on_rollout_done(
            self,
            optim_algo: Algo,
            step: int,
            info: InfoDict,
    ):
        scheduler_values = self.collect_scheduler_values(self.rollout_schedulers)
        self._on_rollout_done(optim_algo, step, info, scheduler_values)

    def on_optimization_done(
            self,
            optim_algo: Algo,
            step: int,
            info: InfoDict,
    ):
        scheduler_values = self.collect_scheduler_values(self.optimization_schedulers)
        self._on_optimization_done(optim_algo, step, info, scheduler_values)

    @staticmethod
    def collect_scheduler_values(schedulers: SchedulerDict) -> dict[str, Any]:
        return {
            scheduler_id: scheduler.step()
            for scheduler_id, scheduler
            in schedulers.items()
        }
