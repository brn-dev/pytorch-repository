from torch import nn

from src.networks.net import Net
from src.networks.layered_net import LayeredNet
from src.networks.net_list import NetList, NetListLike
from src.utils import all_none_or_all_not_none, one_not_none


class SeqNet(LayeredNet):

    @staticmethod
    def compute_sequential_layer_in_out_sizes(
            layer_sizes: list[int] = None,

            in_size: int = None,
            out_sizes: list[int] = None,

            num_layers: int = None,
            num_features: int = None
    ) -> list[tuple[int, int]]:
        parameter_choices = [layer_sizes, in_size, num_layers]

        assert all_none_or_all_not_none(in_size, out_sizes)
        assert one_not_none(parameter_choices), 'only one parameter choice must be used'

        if layer_sizes is not None:
            layers_in_out_sizes = [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

        elif in_size is not None:
            layers_in_out_sizes = []
            for out_size in out_sizes:
                layers_in_out_sizes.append((in_size, out_size))
                in_size = out_size

        elif num_layers is not None:
            layers_in_out_sizes = [(num_features, num_features) for _ in range(num_layers)]

        else:
            raise Exception('This should not happen')

        return layers_in_out_sizes


    @staticmethod
    def from_layer_provider(
            layer_provider: LayeredNet.LayerProvider,
            layers_in_out_sizes: list[tuple[int, int]] = None,
            layers_sizes: list[int] = None,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None
    ) -> 'SeqNet':
        if layers_in_out_sizes is None:
            layers_in_out_sizes = SeqNet.compute_sequential_layer_in_out_sizes(
                layer_sizes=layers_sizes,
                in_size=in_size,
                out_sizes=out_sizes,
                num_layers=num_layers,
                num_features=num_features
            )

        layers = SeqNet.create_layer_list(layer_provider, layers_in_out_sizes)
        return SeqNet(layers)


    @staticmethod
    def find_sequential_in_out_features(layers: NetList):
        in_features, out_features = Net.IN_FEATURES_ANY, Net.OUT_FEATURES_SAME

        for layer in layers:

            if layer.in_features_defined and out_features != Net.OUT_FEATURES_SAME \
                    and layer.in_features != out_features:
                raise ValueError(f'Layer {layer} expects {layer.in_features} features but'
                                 f' it is receiving {out_features}')
            if layer.in_features_defined and in_features == Net.IN_FEATURES_ANY:
                in_features = layer.in_features
            if layer.out_features_defined:
                out_features = layer.out_features

        return in_features, out_features


    def __init__(
            self,
            layers: NetListLike | nn.Sequential,
            allow_undefined_in_out_features: bool = False,
    ):
        if isinstance(layers, nn.Sequential):
            net_layers = []
            for nn_layer in layers:
                net_layers.append(Net.as_net(nn_layer))
            layers = NetList(net_layers)

        in_features, out_features = self.find_sequential_in_out_features(layers)

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            layers=layers,
            layer_connections=LayeredNet.LayerConnections.by_name('sequential', len(layers)),
            allow_undefined_in_out_features=allow_undefined_in_out_features,
        )


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



