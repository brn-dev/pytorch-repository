import abc

import torch


class RLBase(abc.ABC):

    @abc.abstractmethod
    def find_optimal_policy(self, num_episodes: int, *args, **kwargs):
        raise NotImplemented

    @staticmethod
    def compute_returns(rewards: list[float], gamma: float, normalize_returns=False) -> torch.Tensor:
        returns: list[float] = [0] * len(rewards)

        discounted_return = 0.0
        for i in range(len(rewards) - 1, -1, -1):
            discounted_return = rewards[i] + gamma * discounted_return
            returns[i] = discounted_return

        returns_tensor = torch.tensor(returns)

        if normalize_returns:
            returns_tensor = RLBase.normalize_tensor(returns_tensor)

        return returns_tensor

    @staticmethod
    def normalize_tensor(tensor: torch.Tensor):
        return (tensor - tensor.mean()) / (tensor.std() + 1e-6)
