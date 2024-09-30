from torch import optim

def get_lr(optimizer: optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
