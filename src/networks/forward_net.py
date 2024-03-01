from src.networks.net import LayeredNet, LayerList


class ForwardNet(LayeredNet):

    @staticmethod
    def compute_sequential_layer_in_out_sizes(
            layer_sizes: list[int] = None,

            in_size: int = None,
            out_sizes: list[int] = None,

            num_layers: int = None,
            num_features: int = None
    ) -> list[tuple[int, int]]:
        parameter_choices = [layer_sizes, in_size, num_layers]

        assert in_size is not None and out_sizes is not None or in_size is None and out_sizes is None
        assert parameter_choices.count(None) == len(parameter_choices) - 1, 'only one parameter choice can be used'

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
    ) -> 'ForwardNet':
        if layers_in_out_sizes is None:
            layers_in_out_sizes = ForwardNet.compute_sequential_layer_in_out_sizes(
                layer_sizes=layers_sizes,
                in_size=in_size,
                out_sizes=out_sizes,
                num_layers=num_layers,
                num_features=num_features
            )
        layers = ForwardNet.create_layer_list(layer_provider, layers_in_out_sizes)

        return ForwardNet(layers)


    def __init__(self, layers: LayerList):
        super().__init__(
            in_features=layers[0].in_features,
            out_features=layers[-1].out_features,
            layers=layers,
            layer_connections=LayeredNet.LayerConnections.by_name('sequential', len(layers))
        )


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x



