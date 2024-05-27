from typing import Iterator, Union, Iterable, Dict, Callable

from torch import nn

from src.networks.core.net import Net

NetListLike = Union['NetList', Iterable[Net]]


class NetList(nn.ModuleList):

    _modules: Dict[str, Net]

    def __init__(self, layers: Iterable[Net]):
        assert all(isinstance(l, Net) for l in layers)
        super().__init__(modules=layers)

    def __getitem__(self, idx: int | slice) -> Union[Net, 'NetList']:
        if isinstance(idx, slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_string_index(idx)]

    def __iter__(self) -> Iterator[Net]:
        return iter(self._modules.values())

    def all_match(self, condition: Callable[[Net], bool]):
        return all(condition(net) for net in self)

    @classmethod
    def as_net_list(cls, net_list_like: NetListLike) -> 'NetList':
        if isinstance(net_list_like, NetList):
            return net_list_like
        return cls(net_list_like)
