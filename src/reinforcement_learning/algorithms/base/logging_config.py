from dataclasses import dataclass


@dataclass
class LoggingConfig:
    log_reset_info: bool = False
    log_rollout_infos: bool = False
    log_rollout_action_stds: bool = False
    log_last_obs: bool = False

    def __post_init__(self):
        assert not self.log_rollout_action_stds or self.log_rollout_infos, \
            'log_rollout_infos has to be enabled for log_rollout_stds'
