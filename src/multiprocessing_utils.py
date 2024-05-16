from typing import Callable


class CloudpickleFunctionWrapper:

    def __init__(self, fn: Callable):
        self.fn = fn

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.fn)

    def __setstate__(self, ob):
        import cloudpickle
        self.fn = cloudpickle.loads(ob)

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
