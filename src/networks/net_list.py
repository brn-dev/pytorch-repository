from typing import Iterator, Union

from torch import nn

from src.networks.net import Net


class NetList(nn.ModuleList):

    def __init__(self, layers: list[Net]):
        super().__init__(modules=layers)
        self.layers = layers

    def __getitem__(self, idx: int | slice) -> Union[Net, 'NetList']:
        if isinstance(idx, slice):
            return self.__class__(self.layers[idx])
        else:
            return self.layers[idx]

    def __iter__(self) -> Iterator[Net]:
        return iter(self.layers)
