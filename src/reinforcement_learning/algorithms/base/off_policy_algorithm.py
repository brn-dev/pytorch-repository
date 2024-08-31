from typing import TypeVar, Optional

from overrides import override

from src.reinforcement_learning.algorithms.base.base_algorithm import BaseAlgorithm, Policy, LogConf
from src.reinforcement_learning.core.action_noise import ActionNoise
from src.reinforcement_learning.core.buffers.replay.replay_buffer import ReplayBuffer
from src.reinforcement_learning.core.callback import Callback

ReplayBuf = TypeVar('ReplayBuf', bound=ReplayBuffer)

class OffPolicyAlgorithm(BaseAlgorithm[Policy, ReplayBuf, LogConf]):

    def __init__(
            self,
            action_noise: Optional[ActionNoise],
            callback: Callback,
    ):
        self.action_noise = action_noise
        self.callback = callback



