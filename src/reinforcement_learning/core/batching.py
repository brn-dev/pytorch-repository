import numpy as np
import torch


def batched(batch_size: int, *tensors: torch.Tensor, shuffle: bool = False):
    assert len(tensors) > 0

    tensor_length = tensors[0].shape[0]

    assert all(t.shape[0] == tensor_length for t in tensors)

    if shuffle:
        indices = np.random.permutation(tensor_length)
    else:
        indices = np.arange(tensor_length)

    for i in range(0, tensor_length, batch_size):

        batch_tensors: list[torch.Tensor] = []

        for tensor in tensors:
            batch_tensors.append(tensor[indices[i:i + batch_size]])

        yield tuple(batch_tensors)
