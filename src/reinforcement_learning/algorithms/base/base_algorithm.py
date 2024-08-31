import abc
from typing import TypeVar, Callable, Generic, Self

from src.reinforcement_learning.algorithms.base.logging_config import LoggingConfig
from src.reinforcement_learning.core.buffers.base_buffer import BaseBuffer
from src.reinforcement_learning.core.policies.base_policy import BasePolicy

LogConf = TypeVar('LogConf', bound=LoggingConfig)

Buffer = TypeVar('Buffer', bound=BaseBuffer)

Policy = TypeVar('Policy', bound=BasePolicy)
PolicyProvider = Callable[[], Policy]

class BaseAlgorithm(Generic[Policy, Buffer, LogConf], abc.ABC):

    def learn(self, total_timesteps: int) -> Self:
        raise NotImplemented
