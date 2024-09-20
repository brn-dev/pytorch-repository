from torch import nn


class BaseModule(nn.Module):

    def __init__(self):
        super().__init__()

        self.train(False)
        self.train_mode = False
        self.trainable = True


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
