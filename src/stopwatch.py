import time


class Stopwatch:

    def __init__(self):
        self.last_time = time.time()

    def time_passed(self):
        return time.time() - self.last_time

    def reset(self):
        time_passed = self.time_passed()
        self.last_time += time_passed
        return time_passed

