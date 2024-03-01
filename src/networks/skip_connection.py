from typing import Mapping

import numpy as np
import torch
from torch import nn

from src.networks.net import Net, LayeredNet
from src.networks.weighing import WeighingBase, WeighingTypes, WeighingTrainableChoices

class ResidualSkipConnection(LayeredNet):

    @property
    def layers(self) -> Mapping[int, Net]:
        pass

    @property
    def layer_connections(self) -> np.ndarray:
        return self._connections

    @property
    def in_features(self) -> int:
        return self.num_features

    @property
    def out_features(self) -> int:
        return self.num_features

    def __init__(
            self,
            layer: nn.Module,
            num_features: int,

            layer_out_weight: WeighingTypes = 1.0,
            layer_out_weight_trainable: WeighingTrainableChoices = False,

            skip_connection_weight: WeighingTypes = 1.0,
            skip_connection_weight_trainable: WeighingTrainableChoices = False,

            dropout_p: float = 0.0,
            normalization_provider: Net.Provider = None,
    ):
        super().__init__()

        self.layer = layer
        self.num_features = num_features

        self.weigh_skip_connection = \
            WeighingBase.to_weighing(skip_connection_weight, num_features, skip_connection_weight_trainable)
        self.weigh_layer_out = \
            WeighingBase.to_weighing(layer_out_weight, num_features, layer_out_weight_trainable)

        self.dropout = self.provide_dropout(dropout_p) or nn.Identity()
        self.norm = self.provide(normalization_provider, num_features) or nn.Identity()


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

