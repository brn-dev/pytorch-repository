import torch
from torch import nn

from src.networks.nn_base import NNBase
from src.networks.weighing import WeighingBase, WeighingTypes, WeighingTrainableChoices

class ResConnection(NNBase):

    def __init__(
            self,
            layer: nn.Module,
            num_features: int,

            layer_out_weight: WeighingTypes = 1.0,
            layer_out_weight_trainable: WeighingTrainableChoices = False,

            skip_connection_weight: WeighingTypes = 1.0,
            skip_connection_weight_trainable: WeighingTrainableChoices = False,

            dropout_p: float = 0.0,
            normalization_provider: NNBase.Provider = None,
    ):
        super().__init__()

        self.layer = layer

        self.weigh_skip_connection = \
            WeighingBase.to_weighing(skip_connection_weight, num_features, skip_connection_weight_trainable)
        self.weigh_layer_out = \
            WeighingBase.to_weighing(layer_out_weight, num_features, layer_out_weight_trainable)

        self.dropout = NNBase.provide_dropout(dropout_p) or nn.Identity()
        self.norm = NNBase.provide(normalization_provider, num_features) or nn.Identity()


    def forward(self, x, *args, **kwargs):
        layer_out = self.layer(x, *args, **kwargs)
        return self.norm(self.dropout(
            self.weigh_layer_out(layer_out) + self.weigh_skip_connection(x)
        ))

class DenseConnection(NNBase):

    def __init__(
            self,
            layer: nn.Module,
            dropout_p: float = 0.0,
    ):
        super().__init__()

        self.layer = layer
        self.dropout = NNBase.provide_dropout(dropout_p) or nn.Identity()

    def forward(self, x, *args, **kwargs):
        layer_out = self.layer(x, *args, **kwargs)
        return self.dropout(torch.cat(
            (layer_out, x),
            dim=-1
        ))

