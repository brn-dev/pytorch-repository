from torch import nn

from src.hyper_parameters import HasHyperParameters, HyperParameters
from src.module_analysis import count_parameters


class BaseRLModule(nn.Module, HasHyperParameters):

    def __init__(self):
        super().__init__()

        self.train(False)
        self.train_mode = False
        self.trainable = True

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'parameter_count': count_parameters(self)
        })

    def set_train_mode(self, mode: bool) -> None:
        if not self.trainable and mode:
            raise ValueError('Can not set train_mode = True while trainable == False')

        self.train(mode)
        self.train_mode = mode

    def set_trainable(self, trainable: bool):
        self.trainable = trainable
        if not trainable:
            self.set_train_mode(False)

        for param in self.parameters():
            param.requires_grad = trainable
