from dataclasses import dataclass, asdict


@dataclass(init=True)
class HyperParameters:

    def __set_item__(self, key, item):
        raise TypeError('HyperParameters are frozen, cannot set values')

    def __getitem__(self, key: str):
        return getattr(self, key)

    def to_dict(self):
        return asdict(self)
