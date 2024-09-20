import torch
from torch import optim

AUTO_TORCH_DEVICE = 'auto'

TorchDevice = torch.device | str


def get_torch_device(device: TorchDevice | None = None):
    if device is None or device == AUTO_TORCH_DEVICE:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def set_default_torch_device(device: TorchDevice | None = None):
    if device is None:
        device = get_torch_device()
    torch.set_default_device(device)
    return torch.device(device)


def optimizer_to_device(optimizer: optim.Optimizer, device: TorchDevice):
    for state_param in optimizer.state.values():
        tensor_or_dict_to_device(state_param, device)
    return optimizer


def tensor_or_dict_to_device(val: torch.Tensor | dict, device: TorchDevice):
    if isinstance(val, torch.Tensor):
        tensor_to_device(val, device)
    elif isinstance(val, dict):
        dict_to_device(val, device)
    else:
        raise ValueError(f'Unknown type of key {val}: {type(val)}')


def dict_to_device(tensor_dict: dict[str, torch.Tensor | dict], device: TorchDevice) -> None:
    for v in tensor_dict.values():
        tensor_or_dict_to_device(v, device)


def tensor_to_device(tensor: torch.Tensor, device: TorchDevice) -> None:
    tensor.data = tensor.data.to(device)
    if tensor._grad is not None:
        tensor._grad.data = tensor._grad.to(device)
