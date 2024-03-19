from torch import nn

from src.networks.core.net import Net
from src.networks.core.tensor_shape import TensorShape


class LinearNet(Net, nn.Linear):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            device=None,
            dtype=None
    ):
        Net.__init__(
            self,
            in_shape=TensorShape(features=in_features),
            out_shape=TensorShape(features=out_features),
        )
        nn.Linear.__init__(
            self,
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
