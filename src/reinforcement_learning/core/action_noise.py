import abc
import copy
from typing import Optional, Iterable

import numpy as np


class ActionNoise(abc.ABC):

    def reset(self, env_indices: Optional[Iterable[int]] = None):
        pass

    @abc.abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError

class NormalActionNoise(ActionNoise):

    def __init__(
            self,
            mean: np.ndarray,
            std: np.ndarray,
            dtype: np.dtype = np.float32
    ):
        super().__init__()

        self.mean = mean
        self.std = std

        self.dtype = dtype

    def __call__(self) -> np.ndarray:
        return np.random.normal(self.mean, self.std).astype(self.dtype)

class OrnsteinUhlenbeckActionNoise(ActionNoise):

    def __init__(
            self,
            mean: np.ndarray,
            std: np.ndarray,
            theta: float = 0.15,
            dt: float = 1e-2,
            initial_noise: Optional[np.ndarray] = None,
            dtype: np.dtype = np.float32,
    ):
        super().__init__()
        self.mean = mean
        self.std = std
        self.theta = theta
        self.dt = dt
        self.initial_noise = initial_noise
        self.prev_noise = np.zeros_like(self.mean)
        self.dtype = dtype
        self.reset()

    def reset(self, env_indices: Optional[Iterable[int]] = None):
        self.prev_noise = self.initial_noise if self.initial_noise is not None else np.zeros_like(self.mean)

    def __call__(self) -> np.ndarray:
        noise = (
            self.prev_noise
            + self.theta * (self.mean - self.prev_noise) * self.dt
            + self.std * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.prev_noise = noise
        return noise.astype(self.dtype)


class VectorizedActionNoise(ActionNoise):

    def __init__(self, base_noise: ActionNoise, num_envs: int):
        assert num_envs > 0

        self.base_noise = base_noise
        self.num_envs = num_envs
        self.noises = [copy.deepcopy(self.base_noise) for _ in range(num_envs)]

        self.reset()

    def reset(self, env_indices: Optional[Iterable[int]] = None):
        if env_indices is None:
            env_indices = range(self.num_envs)

        for env_index in env_indices:
            self.noises[env_index].reset()

    def __call__(self):
        return np.stack([noise() for noise in self.noises])



