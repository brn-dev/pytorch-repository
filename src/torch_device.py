import torch

def get_torch_device(device: torch.device | str = None):
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

def set_default_torch_device(device: torch.device | str = None):
    if device is None:
        device = get_torch_device()
    torch.set_default_device(device)
