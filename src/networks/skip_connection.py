import torch
from torch import nn

from src.networks.core.layered_net import LayeredNet
from src.networks.core.net import Net
from src.networks.weighing import Weighing, WeighingTypes, WeighingTrainableChoices

class AdditiveSkipConnection(LayeredNet):

    def __init__(
            self,
            layer: Net,

            layer_out_weight: WeighingTypes = 1.0,
            layer_out_weight_trainable: WeighingTrainableChoices = False,

            skip_connection_weight: WeighingTypes = 1.0,
            skip_connection_weight_trainable: WeighingTrainableChoices = False,
    ):
        assert layer.in_features_defined()
        assert layer.in_out_features_same()

        super().__init__(
            in_features=layer.in_features,
            out_features='same',
            layers=[layer],
            layer_connections=[
                (0, 0),
                (0, 1),
                (1, 1)
            ]
        )

        self.layer = layer

        self.weigh_skip_connection = \
            Weighing.to_weighing(skip_connection_weight, layer.in_features, skip_connection_weight_trainable)
        self.weigh_layer_out = \
            Weighing.to_weighing(layer_out_weight, layer.in_features, layer_out_weight_trainable)


    def forward(self, x, *args, **kwargs):
        layer_out = self.layer(x, *args, **kwargs)
        return self.norm(self.dropout(
            self.weigh_layer_out(layer_out) + self.weigh_skip_connection(x)
        ))

class DenseSkipConnection(LayeredNet):

    def __init__(
            self,
            layer: nn.Module,
            dropout_p: float = 0.0,
    ):
        super().__init__()

        self.layer = layer
        self.dropout = self.provide_dropout(dropout_p) or nn.Identity()

    def forward(self, x, *args, **kwargs):
        layer_out = self.layer(x, *args, **kwargs)
        return self.dropout(torch.cat(
            (layer_out, x),
            dim=-1
        ))

