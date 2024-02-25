# Normalization Layers in PyTorch (2.2)

## nn.BatchNorm1d

- Applies Batch Normalization over a 2D or 3D input as described in the paper "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift".

## nn.BatchNorm2d

- Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs with additional channel dimension) as described in the paper "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift".

## nn.BatchNorm3d

- Applies Batch Normalization over a 5D input (a mini-batch of 3D inputs with additional channel dimension) as described in the paper "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift".

## nn.LazyBatchNorm1d

- A `torch.nn.BatchNorm1d` module with lazy initialization of the `num_features` argument of the BatchNorm1d that is inferred from the `input.size(1)`.

## nn.LazyBatchNorm2d

- A `torch.nn.BatchNorm2d` module with lazy initialization of the `num_features` argument of the BatchNorm2d that is inferred from the `input.size(1)`.

## nn.LazyBatchNorm3d

- A `torch.nn.BatchNorm3d` module with lazy initialization of the `num_features` argument of the BatchNorm3d that is inferred from the `input.size(1)`.

## nn.GroupNorm

- Applies Group Normalization over a mini-batch of inputs.

## nn.SyncBatchNorm

- Applies Batch Normalization over a N-Dimensional input (a mini-batch of [N-2]D inputs with additional channel dimension) as described in the paper "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift".

## nn.InstanceNorm1d

- Applies Instance Normalization.

## nn.InstanceNorm2d

- Applies Instance Normalization.

## nn.InstanceNorm3d

- Applies Instance Normalization.

## nn.LazyInstanceNorm1d

- A `torch.nn.InstanceNorm1d` module with lazy initialization of the `num_features` argument.

## nn.LazyInstanceNorm2d

- A `torch.nn.InstanceNorm2d` module with lazy initialization of the `num_features` argument.

## nn.LazyInstanceNorm3d

- A `torch.nn.InstanceNorm3d` module with lazy initialization of the `num_features` argument.

## nn.LayerNorm

- Applies Layer Normalization over a mini-batch of inputs.

## nn.LocalResponseNorm

- Applies local response normalization over an input signal.