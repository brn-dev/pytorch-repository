import torch


def batched(batch_size: int, *tensors: torch.Tensor):
    assert len(tensors) > 0

    tensor_length = tensors[0].shape[0]

    assert all(t.shape[0] == tensor_length for t in tensors)

    for i in range(0, tensor_length, batch_size):

        batch_tensors: list[torch.Tensor] = []

        for tensor in tensors:
            batch_tensors.append(tensor[i:i + batch_size])

        yield tuple(batch_tensors)
