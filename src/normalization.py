from src.networks.normalization.batch_renorm import BatchRenorm
from src.torch_nn_modules import nn_normalization_classes

normalization_classes = (
    *nn_normalization_classes,
    BatchRenorm
)
