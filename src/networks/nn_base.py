from torch import nn

class NNBase(nn.Module):

    @staticmethod
    def is_dropout_active(dropout_p: float | None):
        return dropout_p is not None and dropout_p > 0
