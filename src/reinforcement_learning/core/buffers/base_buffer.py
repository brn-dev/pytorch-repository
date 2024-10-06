import abc
from typing import TypeVar, NamedTuple, Generic

import numpy as np
import torch
from gymnasium import Env

from src.hyper_parameters import HasHyperParameters, HyperParameters
from src.reinforcement_learning.gym.env_analysis import get_obs_shape, get_action_shape, get_num_envs
from src.tags import HasTags
from src.torch_device import TorchDevice, get_torch_device
from src.reinforcement_learning.core.type_aliases import ShapeDict

BufferSamples = TypeVar('BufferSamples', bound=NamedTuple)


class BaseBuffer(HasHyperParameters, HasTags, Generic[BufferSamples], abc.ABC):

    pos: int
    full: bool

    def __init__(
            self,
            step_size: int,
            num_envs: int,
            obs_shape: tuple[int, ...] | ShapeDict,
            action_shape: tuple[int, ...],
            reward_scale: float,
            torch_device: TorchDevice = 'cpu',
            torch_dtype: torch.dtype = torch.float32,
            np_dtype: np.dtype = np.float32,
    ):
        self.step_size = step_size
        self.num_envs = num_envs

        self.obs_shape = obs_shape
        self.action_shape = action_shape

        self._reward_scale = float(reward_scale) if reward_scale != 1 else None

        self.torch_device = get_torch_device(torch_device)
        self.torch_dtype = torch_dtype

        self.np_dtype = np_dtype

        self.reset()

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'step_size': self.step_size,
            'num_envs': self.num_envs,
            'total_size': self.step_size * self.num_envs,
            'reward_scale': self.reward_scale,
            'torch_device': str(self.torch_device),
            'torch_dtype': str(self.torch_dtype),
            'np_dtype': str(self.np_dtype),
        })

    @property
    def reward_scale(self) -> float:
        return self._reward_scale if self._reward_scale is not None else 1.0

    @property
    def size(self):
        return self.step_size if self.full else self.pos

    def reset(self):
        self.pos = 0
        self.full = False

    def extend(self, *args) -> None:
        for sample in zip(*args):
            self.add(*sample)

    @abc.abstractmethod
    def add(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def scale_rewards(self, reward: np.ndarray) -> np.ndarray:
        scale = self._reward_scale
        if scale is not None:
            return reward * scale
        return reward

    def unscale_rewards(self, scaled_reward: np.ndarray) -> np.ndarray:
        scale = self._reward_scale
        if scale is not None:
            return scaled_reward / scale
        return scaled_reward

    def to_torch(self, data: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data, device=self.torch_device, dtype=self.torch_dtype)

    def all_to_torch(self, arrays: tuple[np.ndarray, ...]) -> tuple[torch.Tensor, ...]:
        return tuple(map(self.to_torch, arrays))

    @staticmethod
    def flatten(arr: np.ndarray) -> np.ndarray:
        shape = arr.shape
        return arr.reshape(shape[0] * shape[1], *shape[2:])

    @classmethod
    def for_env(
            cls,
            env: Env,
            buffer_size: int,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype = torch.float32,
            np_dtype: np.dtype = np.float32,
            **buffer_kwargs
    ):
        num_envs = get_num_envs(env)

        obs_shape = get_obs_shape(env)
        action_shape = get_action_shape(env)

        # noinspection PyArgumentList
        return cls(
            step_size=buffer_size,
            num_envs=num_envs,
            obs_shape=obs_shape,
            action_shape=action_shape,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
            np_dtype=np_dtype,
            **buffer_kwargs
        )
