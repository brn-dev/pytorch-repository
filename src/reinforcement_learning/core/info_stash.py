from dataclasses import dataclass
from typing import Any, Literal

from src.reinforcement_learning.core.infos import InfoDict, stack_infos, concat_infos, combine_infos
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

class InfoBuffer:

    infos: list[InfoDict]

    def __init__(self, enabled: bool = True):
        self.reset(enabled)

    def reset(self, enabled: bool = True):
        self.infos = create_stash_list(enabled)

    def append(self, info: InfoDict):
        self.infos.append(info)

    def stack_infos(self):
        return combine_infos(self.infos, 'stack')

    def concat_infos(self):
        return combine_infos(self.infos, 'concat')

    def combine_infos(self, combination_method: Literal['stack', 'concat']) -> InfoDict:
        return combine_infos(self.infos, combination_method)
