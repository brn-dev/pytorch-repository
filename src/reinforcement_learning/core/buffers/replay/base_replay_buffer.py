import abc

from src.reinforcement_learning.core.buffers.base_buffer import BaseBuffer, BufferSamples


class BaseReplayBuffer(BaseBuffer[BufferSamples], abc.ABC):

    @abc.abstractmethod
    def sample(self, batch_size: int, with_replacement: bool = False) -> BufferSamples:
        raise NotImplemented
