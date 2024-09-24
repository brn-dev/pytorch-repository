import abc
import itertools
from typing import TypeVar, Callable, Generic, Self, Iterable

import gymnasium
import numpy as np
import torch
from torch import nn

from src.console import print_warning
from src.hyper_parameters import HasHyperParameters, HyperParameters
from src.module_analysis import count_parameters
from src.reinforcement_learning.core.action_selectors.continuous_action_selector import ContinuousActionSelector
from src.reinforcement_learning.core.buffers.base_buffer import BaseBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.infos import InfoDict
from src.reinforcement_learning.core.logging import LoggingConfig
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.gym.singleton_vector_env import as_vec_env
from src.torch_device import TorchDevice, get_torch_device

LogConf = TypeVar('LogConf', bound=LoggingConfig)

Buffer = TypeVar('Buffer', bound=BaseBuffer)

Policy = TypeVar('Policy', bound=BasePolicy)
PolicyProvider = Callable[[], Policy]


class BaseAlgorithm(HasHyperParameters, Generic[Policy, Buffer, LogConf], abc.ABC):

    def __init__(
            self,
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            buffer: Buffer,
            gamma: float,
            sde_noise_sample_freq: int | None,
            callback: Callback,
            logging_config: LogConf,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype,
    ):

        self.env, self.num_envs = as_vec_env(env)
        self.observation_space: gymnasium.spaces.Space = self.env.get_wrapper_attr('single_observation_space')
        self.action_space: gymnasium.spaces.Space = self.env.get_wrapper_attr('single_action_space')

        self.policy: Policy = (policy if isinstance(policy, BasePolicy) else policy()).to(torch_device)
        self.buffer = buffer

        self.gamma = gamma

        if not policy.uses_sde and sde_noise_sample_freq is not None:
            print_warning(f' SDE noise sample freq is set to {sde_noise_sample_freq} despite not using SDE \n')
        if policy.uses_sde and sde_noise_sample_freq is None:
            raise ValueError(f'SDE noise sample freq is set to None despite using SDE')

        self.sde_noise_sample_freq = sde_noise_sample_freq

        self.logging_config = logging_config
        if (self.logging_config.log_rollout_action_stds
                and not isinstance(policy.actor.action_selector, ContinuousActionSelector)):
            raise ValueError('Cannot log action distribution stds with non continuous action selector')

        self.callback = callback

        self.torch_device = get_torch_device(torch_device)
        self.torch_dtype = torch_dtype

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'env': str(self.env),
            'num_envs': self.num_envs,
            'policy': self.policy.collect_hyper_parameters(),
            'policy_parameter_count': count_parameters(self.policy),
            'policy_repr': str(self.policy),
            'buffer': self.get_hps_or_str(self.buffer),
            'buffer_step_size': self.buffer.buffer_size,
            'buffer_total_size': self.buffer.buffer_size * self.num_envs,
            'gamma': self.gamma,
            'sde_noise_sample_freq': self.sde_noise_sample_freq,
            'torch_device': str(self.torch_device),
            'torch_dtype': str(self.torch_dtype),
        })

    @abc.abstractmethod
    def optimize(
            self,
            last_obs: np.ndarray,
            last_episode_starts: np.ndarray,
            info: InfoDict
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def perform_rollout(
            self,
            max_steps: int,
            obs: np.ndarray,
            episode_starts: np.ndarray,
            info: InfoDict
    ) -> tuple[int, np.ndarray, np.ndarray]:
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def _should_optimize(self):
        # Used in off policy to delay training until a set amount of steps have been collected
        return True

    def learn(self, total_timesteps: int) -> Self:
        obs, _ = self.env.reset()
        episode_starts = np.ones(self.num_envs, dtype=bool)

        step = 0
        while step < total_timesteps:
            info: InfoDict = {}

            self.policy.set_train_mode(False)
            with torch.no_grad():
                steps_performed, obs, episode_starts = self.perform_rollout(
                    max_steps=total_timesteps - step,
                    obs=obs,
                    episode_starts=episode_starts,
                    info=info
                )
            step += steps_performed
            self.callback.on_rollout_done(self, step, info)

            if self._should_optimize():
                self.policy.set_train_mode(True)
                self.optimize(obs, episode_starts, info)

                self.callback.on_optimization_done(self, step, info)

        return self

    def normalize_actions(self, actions: np.ndarray) -> np.ndarray:
        assert isinstance(self.action_space, gymnasium.spaces.Box), 'Can only normalize continuous actions'
        low, high = self.action_space.low, self.action_space.high
        return 2.0 * ((actions - low) / (high - low)) - 1.0

    def denormalize_actions(self, normalized_actions: np.ndarray) -> np.ndarray:
        assert isinstance(self.action_space, gymnasium.spaces.Box), 'Can only denormalize continuous actions'
        low, high = self.action_space.low, self.action_space.high
        return low + (0.5 * (normalized_actions + 1.0) * (high - low))

    def clip_actions(self, actions: np.ndarray) -> np.ndarray:
        assert isinstance(self.action_space, gymnasium.spaces.Box), 'Can only clip continuous actions'
        low, high = self.action_space.low, self.action_space.high
        return np.clip(actions, low, high)

    def to_tensor(self, arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr, dtype=self.torch_dtype, device=self.torch_device)

    @staticmethod
    def chain_parameters(*modules: nn.Module) -> Iterable[torch.Tensor]:
        return itertools.chain(*[
            module.parameters() for module in modules
        ])
