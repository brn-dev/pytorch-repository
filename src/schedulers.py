from typing import Generic, TypeVar

T = TypeVar('T')

class ValueScheduler(Generic[T]):

    def __init__(self, schedule: dict[int, T], keep_last_value: bool):
        assert all(k >= 0 for k in schedule.keys())
        assert 0 in schedule

        self.schedule = schedule
        self.keep_last_value = keep_last_value

        self.step_counter = 0
        self.last_value: T | None = None

    def step(self):
        next_value = self.schedule.get(self.step_counter, None)
        self.step_counter += 1

        if next_value is None and self.keep_last_value:
            next_value = self.last_value

        self.last_value = next_value

        return next_value
