from typing import Iterator, Union, Iterable

from torch import nn

from src.networks.net import Net

NetListLike = Union['NetList', Iterable[Net]]

class NetList(nn.ModuleList):


    def __init__(self, layers: Iterable[Net]):
        super().__init__(modules=layers)
        self.layers = list(layers)

    def __getitem__(self, idx: int | slice) -> Union[Net, 'NetList']:
        if isinstance(idx, slice):
            return self.__class__(self.layers[idx])
        else:
            return self.layers[idx]

    def __iter__(self) -> Iterator[Net]:
        return iter(self.layers)

    @classmethod
    def as_net_list(cls, net_list_like: NetListLike) -> 'NetList':
        if isinstance(net_list_like, NetList):
            return net_list_like
        return cls(net_list_like)
