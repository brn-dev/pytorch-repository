from dataclasses import dataclass
from typing import Any

from src.reinforcement_learning.core.infos import InfoDict
from src.void_list import VoidList


@dataclass
class LoggingConfig:
    log_rollout_infos: bool = False
    log_rollout_action_stds: bool = False
    log_last_obs: bool = False

    def __post_init__(self):
        assert not self.log_rollout_action_stds or self.log_rollout_infos, \
            'log_rollout_infos has to be enabled for log_rollout_stds'



def create_log_list(log_enabled: bool) -> list:
    if log_enabled:
        return []
    return VoidList()

def log_if_enabled(info: InfoDict, key: str, value: Any, log_enabled: bool):
    if log_enabled:
        info[key] = value
