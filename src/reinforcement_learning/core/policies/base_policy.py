import json
import os
from pathlib import Path
from typing import Optional

import torch

from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.policies.components.actor import Actor
from src.reinforcement_learning.core.policies.components.base_component import BasePolicyComponent
from src.reinforcement_learning.core.policies.components.feature_extractors import FeatureExtractor, IdentityExtractor
from src.reinforcement_learning.core.type_aliases import TensorObs
from src.tags import Tags


class BasePolicy(BasePolicyComponent):

    train_mode: bool

    def __init__(
            self,
            actor: Actor,
            shared_feature_extractor: Optional[FeatureExtractor] = None
    ):
        super().__init__(shared_feature_extractor or IdentityExtractor())
        self.actor = actor

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'actor': self.actor.collect_hyper_parameters(),
        })

    def collect_tags(self) -> Tags:
        return self.combine_tags(super().collect_tags(), self.actor.collect_tags())

    @property
    def shared_feature_extractor(self):
        return self.feature_extractor

    @property
    def uses_sde(self):
        return self.actor.uses_sde

    def act(self, obs: TensorObs) -> ActionSelector:
        obs = self.shared_feature_extractor(obs)
        return self.actor(obs)

    def reset_sde_noise(self, batch_size: int) -> None:
        self.actor.reset_sde_noise(batch_size)

    def as_actor_policy(self):
        from src.reinforcement_learning.core.policies.actor_policy import ActorPolicy
        return ActorPolicy(self.actor, self.shared_feature_extractor)

    def save(self, file_path: str, as_state_dict: bool = True, save_meta_data: bool = True, **meta_data):
        file_path = Path(file_path)

        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if as_state_dict:
            assert file_path.suffix == '.pth', 'state dicts must be saved as a .pth file'
            torch.save(self.state_dict(), file_path)
        else:
            assert file_path.suffix == '.pt', 'policy must be saved as a .pt file'
            torch.save(self, file_path)
        
        if save_meta_data:
            with open(file_path.with_suffix('.meta.json'), 'w') as f:
                json.dump({
                    'hyper_parameters': self.collect_hyper_parameters(),
                    'repr': repr(self),
                    'policy_path': str(file_path.absolute()),
                    **meta_data
                }, f)
