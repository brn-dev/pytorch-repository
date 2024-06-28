from overrides import override

import torch

from src.networks.core.layer_connections import LayerConnections
from src.networks.core.layered_net import LayeredNet, LayerProvider
from src.networks.core.net import Net


class DenseSkipConnection(LayeredNet):

    def __init__(
            self,
            layer: Net,
    ):
        super().__init__(
            layers=[layer],
            layer_connections=LayerConnections.by_name('full', 1),
            feature_combination_method='dense'
        )

        self.layer = layer

    def forward(self, x, *args, **kwargs):
        layer_out = self.layer(x, *args, **kwargs)
        return torch.cat(
            (layer_out, x),
            dim=-1
        )

    @classmethod
    @override
    def from_layer_provider(
            cls,
            layer_provider: LayerProvider,
            in_features: int,
            out_features: int,
    ) -> 'DenseSkipConnection':
        return DenseSkipConnection(cls.provide_layer(
            provider=layer_provider,
            layer_nr=0,
            is_last_layer=True,
            in_features=in_features,
            out_features=out_features,
        ))
