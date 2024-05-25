import abc
from typing import Generic, TypeVar, Callable

T = TypeVar('T')


NewValueCallback = Callable[[T], None]

class Scheduler(abc.ABC, Generic[T]):

    def __init__(self, new_value_callback: NewValueCallback | None):
        self.new_value_callback = new_value_callback

    @abc.abstractmethod
    def step(self) -> T:
        raise NotImplemented


class FixedValueScheduler(Scheduler[T]):

    def __init__(
            self,
            schedule: dict[int, T],
            keep_last_value: bool,
            new_value_callback: NewValueCallback | None = None
    ):
        assert all(k >= 0 for k in schedule.keys())
        assert 0 in schedule

        super().__init__(new_value_callback)

        self.schedule = schedule
        self.keep_last_value = keep_last_value

        self.step_counter = 0
        self.last_value: T | None = None

    def step(self) -> T:
        next_value = self.schedule.get(self.step_counter, None)
        self.step_counter += 1

        if next_value is not None and self.new_value_callback is not None:
            self.new_value_callback(next_value)

        if next_value is None and self.keep_last_value:
            next_value = self.last_value

        self.last_value = next_value

        return next_value

class OneStepRecursiveScheduler(Scheduler[T]):

    def __init__(
            self,
            initial_value: T,
            next_value_function: Callable[[T], T],
            new_value_callback: NewValueCallback | None = None
    ):
        super().__init__(new_value_callback)

        self.current_value = initial_value
        self.next_value_function = next_value_function

    def step(self) -> T:
        current_value = self.current_value
        self.current_value = self.next_value_function(current_value)

        if self.new_value_callback is not None:
            self.new_value_callback(current_value)
        return current_value
