import torch.nn as nn

from fnn import FNN


def lecun_initialization(linear: nn.Linear):
    nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='linear')
    nn.init.zeros_(linear.bias)


class SelfNormalizingFNN(FNN):

    def __init__(self, input_size, hidden_sizes, output_size, activate_last_layer=False):
        super().__init__(
            input_size,
            hidden_sizes,
            output_size,
            activation_provider=lambda: nn.SELU(),
            layer_initialization=lecun_initialization,
            activate_last_layer=activate_last_layer
        )

    def forward(self, x):
        x = self.fnn.forward(x)
        return x
