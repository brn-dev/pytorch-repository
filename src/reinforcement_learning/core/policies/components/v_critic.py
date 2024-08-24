import torch
from torch import nn


# State value critic
class VCritic(nn.Module):
    
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)
