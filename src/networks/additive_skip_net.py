import torch
from torch import nn

from src.networks.core.net import Net
from src.networks.forward_net import SeqNet
from src.networks.weighing import WeighingTrainableChoices


class AdditiveSkipNet(SeqNet):

    def __init__(
            self,
            layer_provider: SeqNet.LayerProvider,
            num_layers: int,
            num_features: int = None,

            connections: Net.ConnectionsLike = 'full',

            weights_trainable: WeighingTrainableChoices = False,
            initial_direct_connection_weight: float = 1.0,
            initial_skip_connection_weight: float = 1.0,

            # dropout_p: float = 0.0,
            # normalization_provider: NNBase.Provider = None,
    ):
        super().__init__(layer_provider, num_layers=num_layers, num_features=num_features)

        connections = self.LayerConnections.to_np(connections, num_layers)
        self.connections = connections

        mask = torch.zeros((num_layers + 1, num_features, num_layers + 1))
        weight = torch.zeros((num_layers + 1, num_features, num_layers + 1))

        for from_idx, to_idx in connections:
            mask[to_idx, :, from_idx] = 1.0
            weight[to_idx, :, from_idx] = (initial_direct_connection_weight
                                           if from_idx == to_idx
                                           else initial_skip_connection_weight)

        self.mask = nn.Parameter(mask, requires_grad=False)
        self.weight = nn.Parameter(weight, requires_grad=weights_trainable)

    def forward(self, x: torch.Tensor, return_dense=False, **kwargs):

        dense_tensor = torch.zeros_like(x.float()) \
                .unsqueeze(-1).repeat_interleave(self.num_layers + 1, dim=-1)
        dense_tensor[..., 0] = x

        for i, layer in enumerate(self.layers):
            # TODO: can I take the first i rows instead of all rows?
            layer_input = (dense_tensor * self.mask[i] * self.weight[i]).sum(dim=-1)
            layer_output = layer(layer_input, **kwargs)
            dense_tensor[..., i + 1] = layer_output

        out = (dense_tensor * self.mask[-1] * self.weight[-1]).sum(dim=-1)

        if return_dense:
            return out, dense_tensor
        else:
            return out
