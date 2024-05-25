import abc

import torch
from torch import nn

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.action_selectors.state_dependent_noise_action_selector import \
    StateDependentNoiseActionSelector


class BasePolicy(nn.Module, abc.ABC):

    action_selector: ActionSelector
    uses_sde: bool

    def __init__(
            self,
            network: nn.Module,
            action_selector: ActionSelector
    ):
        super().__init__()
        self.network = network

        self.replace_action_selector(action_selector, copy_action_net_weights=False)

    @abc.abstractmethod
    def process_obs(self, obs: torch.Tensor) -> tuple[ActionSelector, dict[str, torch.Tensor]]:
        raise NotImplemented

    def forward(self, obs: torch.Tensor):
        return self.network(obs)

    def replace_action_selector(self, new_action_selector: ActionSelector, copy_action_net_weights: bool) -> None:
        if copy_action_net_weights:
            new_action_selector.action_net.load_state_dict(self.action_selector.action_net.state_dict())
        self.action_selector = new_action_selector
        self.uses_sde = isinstance(self.action_selector, StateDependentNoiseActionSelector)

    def reset_sde_noise(self, batch_size: int) -> None:
        if self.uses_sde:
            self.action_selector: StateDependentNoiseActionSelector

            self.action_selector.sample_exploration_noise(batch_size)
