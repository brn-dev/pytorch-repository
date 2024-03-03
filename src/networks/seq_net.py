from src.networks.layer_connections import LayerConnections
from src.networks.layered_net import LayeredNet, LayerProvider
from src.networks.net import Net
from src.networks.net_list import NetList
from src.utils import all_none_or_all_not_none, one_not_none


class SeqNet(LayeredNet):

    def __init__(self, layers: NetList, allow_undefined_in_out_features: bool = True):
        in_features, out_features = self.find_sequential_in_out_features(layers)

        super().__init__(
            in_features=in_features,
            out_features=out_features,
            layers=layers,
            layer_connections=LayerConnections.by_name('sequential', len(layers)),
            allow_undefined_in_out_features=allow_undefined_in_out_features,
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


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


    @staticmethod
    def resolve_sequential_in_out_features(
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
            in_out_features = [(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

        elif in_size is not None:
            in_out_features = []
            for out_size in out_sizes:
                in_out_features.append((in_size, out_size))
                in_size = out_size

        elif num_layers is not None:
            in_out_features = [(num_features, num_features) for _ in range(num_layers)]

        else:
            raise Exception('This should not happen')

        return in_out_features


    @staticmethod
    def from_layer_provider(
            layer_provider: LayerProvider,
            in_out_features: list[tuple[int, int]] = None,
            layers_sizes: list[int] = None,
            in_size: int = None,
            out_sizes: list[int] = None,
            num_layers: int = None,
            num_features: int = None
    ) -> 'SeqNet':
        if in_out_features is None:
            in_out_features = SeqNet.resolve_sequential_layer_in_out_sizes(
                layer_sizes=layers_sizes,
                in_size=in_size,
                out_sizes=out_sizes,
                num_layers=num_layers,
                num_features=num_features
            )

        layers = SeqNet.provide_layers(layer_provider, in_out_features)
        return SeqNet(layers)
