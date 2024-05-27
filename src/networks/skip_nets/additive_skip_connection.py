from overrides import override

from src.networks.core.layer_connections import LayerConnections
from src.networks.core.layered_net import LayeredNet, LayerProvider
from src.networks.core.net import Net
from src.networks.weighing import WeighingTypes, WeighingTrainableChoices, Weighing


class AdditiveSkipConnection(LayeredNet):

    def __init__(self, layer: Net):
        super().__init__(
            layers=[layer],
            layer_connections=LayerConnections.by_name('full', 1),
            combination_method='additive',
            require_definite_dimensions=['features'],
        )

        self.layer = layer
        self.num_features = layer.in_shape.get_definite_size('features')

    def forward(self, x, *args, **kwargs):
        layer_out = self.layer(x, *args, **kwargs)
        return layer_out + x

    @classmethod
    @override
    def from_layer_provider(
            cls,
            layer_provider: LayerProvider,
            num_features: int,

            layer_out_weight: WeighingTypes = 1.0,
            layer_out_weight_trainable: WeighingTrainableChoices = False,

            skip_connection_weight: WeighingTypes = 1.0,
            skip_connection_weight_trainable: WeighingTrainableChoices = False,
    ) -> 'AdditiveSkipConnection':
        return AdditiveSkipConnection(
            layer=cls.provide_layer(
                provider=layer_provider,
                layer_nr=0,
                is_last_layer=True,
                in_features=num_features,
                out_features=num_features,
            ),
        )

class WeightedAdditiveSkipConnection(AdditiveSkipConnection):

    def __init__(
            self,
            layer: Net,

            layer_out_weight: WeighingTypes = 1.0,
            layer_out_weight_trainable: WeighingTrainableChoices = False,

            skip_connection_weight: WeighingTypes = 1.0,
            skip_connection_weight_trainable: WeighingTrainableChoices = False,
    ):
        super().__init__(layer=layer)

        self.layer = layer
        self.num_features = layer.in_shape.get_definite_size('features')

        self.weigh_skip_connection = Weighing.to_weighing(
            num_features=self.num_features,
            weight=skip_connection_weight,
            trainable=skip_connection_weight_trainable,
        )
        self.weigh_layer_out = Weighing.to_weighing(
            num_features=self.num_features,
            weight=layer_out_weight,
            trainable=layer_out_weight_trainable,
        )

    def forward(self, x, *args, **kwargs):
        layer_out = self.layer(x, *args, **kwargs)
        return self.weigh_layer_out(layer_out) + self.weigh_skip_connection(x)

    @classmethod
    def weighted_from_layer_provider(
            cls,
            layer_provider: LayerProvider,
            num_features: int,

            layer_out_weight: WeighingTypes = 1.0,
            layer_out_weight_trainable: WeighingTrainableChoices = False,

            skip_connection_weight: WeighingTypes = 1.0,
            skip_connection_weight_trainable: WeighingTrainableChoices = False,
    ) -> 'WeightedAdditiveSkipConnection':
        return WeightedAdditiveSkipConnection(
            layer=cls.provide_layer(
                provider=layer_provider,
                layer_nr=0,
                is_last_layer=True,
                in_features=num_features,
                out_features=num_features,
            ),
            layer_out_weight=layer_out_weight,
            layer_out_weight_trainable=layer_out_weight_trainable,
            skip_connection_weight=skip_connection_weight,
            skip_connection_weight_trainable=skip_connection_weight_trainable,
        )
