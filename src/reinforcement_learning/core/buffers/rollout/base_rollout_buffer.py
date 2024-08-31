import abc
from typing import Generator

from src.reinforcement_learning.core.buffers.base_buffer import BaseBuffer, BufferSamples


class BaseRolloutBuffer(BaseBuffer[BufferSamples], abc.ABC):

    @abc.abstractmethod
    def get_samples(
            self,
            batch_size: int | None = None,
            shuffled: bool = True
    ) -> Generator[BufferSamples, None, None]:
        raise NotImplemented
