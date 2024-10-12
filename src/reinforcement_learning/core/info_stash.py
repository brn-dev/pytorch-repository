from dataclasses import dataclass
from typing import Any

from src.reinforcement_learning.core.infos import InfoDict
from src.void_list import VoidList


@dataclass
class InfoStashConfig:
    stash_rollout_infos: bool = False
    stash_rollout_action_stds: bool = False
    stash_last_obs: bool = False

    def __post_init__(self):
        assert not self.stash_rollout_action_stds or self.stash_rollout_infos, \
            'stash_rollout_infos has to be enabled for stash_rollout_stds'



def create_stash_list(stash_enabled: bool) -> list:
    if stash_enabled:
        return []
    return VoidList()

def stash_if_enabled(info: InfoDict, key: str, value: Any, stash_enabled: bool):
    if stash_enabled:
        info[key] = value
