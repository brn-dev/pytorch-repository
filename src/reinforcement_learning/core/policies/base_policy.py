import abc
import itertools
from typing import Callable, Iterable, Any

import torch
from torch import nn

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.policies.components.actor import Actor
from src.reinforcement_learning.core.type_aliases import TensorObsPreprocessing, TensorObs


class BasePolicy(nn.Module):

    def __init__(
            self,
            actor: Actor,
            obs_preprocessing: TensorObsPreprocessing = nn.Identity()
    ):
        super().__init__()
        self.actor = actor
        self.obs_preprocessing = obs_preprocessing

    # In case of off-policy algorithms, this should only return the parameters of the components which are actually
    # optimized
    def collect_trainable_parameters(self):
        return self.parameters()

    def act(self, obs: TensorObs) -> ActionSelector:
        obs = self.obs_preprocessing(obs)
        return self.actor(obs)

    @property
    def uses_sde(self):
        return self.actor.uses_sde

    @property
    def action_selector(self):
        return self.actor.action_selector

    def replace_action_selector(self, new_action_selector: ActionSelector, copy_action_net_weights: bool) -> None:
        self.actor.replace_action_selector(new_action_selector, copy_action_net_weights)

    def reset_sde_noise(self, batch_size: int) -> None:
        self.actor.reset_sde_noise(batch_size)

    def set_train_mode(self, mode: bool) -> None:
        self.train(mode)

    @staticmethod
    def _chain_trainable_parameters(modules: Iterable[nn.Module | Any]) -> Iterable[torch.Tensor]:
        return itertools.chain(*[
            module.parameters()
            for module in modules
            if isinstance(module, nn.Module)
        ])
