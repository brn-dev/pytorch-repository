import abc
from typing import TypeVar, NamedTuple, Generic

import numpy as np
import torch
from gymnasium import Env

from src.reinforcement_learning.gym.env_analysis import get_obs_shape, get_action_shape, get_num_envs
from src.torch_device import TorchDevice, get_torch_device
from src.reinforcement_learning.core.type_aliases import ShapeDict

BufferSamples = TypeVar('BufferSamples', bound=NamedTuple)


class BaseBuffer(Generic[BufferSamples], abc.ABC):

    def __init__(
            self,
            buffer_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...] | ShapeDict,
            action_shape: tuple[int, ...],
            torch_device: TorchDevice = 'cpu',
            torch_dtype: torch.dtype = torch.float32,
            np_dtype: np.dtype = np.float32,
    ):
        self.buffer_size = buffer_size
        self.num_envs = num_envs

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.torch_device = get_torch_device(torch_device)
        self.torch_dtype = torch_dtype

        self.np_dtype = np_dtype

        self.pos = 0
        self.full = False

    @property
    def size(self):
        return self.buffer_size if self.full else self.pos

    def reset(self):
        self.pos = 0
        self.full = False

    def extend(self, *args) -> None:
        for sample in zip(*args):
            self.add(*sample)

    @abc.abstractmethod
    def add(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def to_torch(self, data: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data, device=self.torch_device, dtype=self.torch_dtype)

    def all_to_torch(self, arrays: tuple[np.ndarray, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(map(self.to_torch, arrays))

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    @classmethod
    def for_env(
            cls,
            env: Env,
            buffer_size: int,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype = torch.float32,
            np_dtype: np.dtype = np.float32,
    ):
        num_envs = get_num_envs(env)

        obs_shape = get_obs_shape(env)
        action_shape = get_action_shape(env)

        return cls(
            buffer_size=buffer_size,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
        )
