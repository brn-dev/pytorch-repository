import abc
import itertools
import json
import os
from typing import TypeVar, Callable, Generic, Self, Iterable, Any

import gymnasium
import joblib
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
from src.reinforcement_learning.core.info_stash import InfoStashConfig
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.gym.env_analysis import get_unique_env_ids, get_unique_env_specs
from src.reinforcement_learning.gym.singleton_vector_env import as_vec_env
from src.tags import HasTags, Tags
from src.torch_device import TorchDevice, get_torch_device


POLICY_FILE_SUFFIX = '.policy.state_dict.pth'
BUFFER_FILE_SUFFIX = '.buffer.pkl'
META_DATA_FILE_SUFFIX = '.meta.json'

StashConf = TypeVar('StashConf', bound=InfoStashConfig)

Buffer = TypeVar('Buffer', bound=BaseBuffer)

Policy = TypeVar('Policy', bound=BasePolicy)
PolicyProvider = Callable[[], Policy]


class BaseAlgorithm(HasHyperParameters, HasTags, Generic[Policy, Buffer, StashConf], abc.ABC):

    def __init__(
            self,
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            buffer: Buffer,
            gamma: float,
            sde_noise_sample_freq: int | None,
            callback: Callback,
            stash_config: StashConf,
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

        self.stash_config = stash_config

        if (self.stash_config.stash_rollout_action_stds
                and not isinstance(policy.actor.action_selector, ContinuousActionSelector)):
            raise ValueError('Cannot log action distribution stds with non continuous action selector')

        self.callback = callback

        self.torch_device = get_torch_device(torch_device)
        self.torch_dtype = torch_dtype

        self.steps_performed = 0

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'env': str(self.env),
            'num_envs': self.num_envs,
            'env_specs': get_unique_env_specs(self.env),
            'policy': self.policy.collect_hyper_parameters(),
            'policy_parameter_count': count_parameters(self.policy),
            'policy_repr': str(self.policy),
            'buffer': self.buffer.collect_hyper_parameters(),
            'gamma': self.gamma,
            'sde_noise_sample_freq': self.sde_noise_sample_freq,
            'torch_device': str(self.torch_device),
            'torch_dtype': str(self.torch_dtype),
        })

    def collect_tags(self) -> Tags:
        return self.combine_tags(
            [type(self).__name__],
            get_unique_env_ids(self.env),
            self.policy.collect_tags(),
            self.buffer.collect_tags(),
            super().collect_tags(),
        )

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
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def _should_optimize(self):
        # Used in off policy to delay training until a set amount of steps have been collected
        return True

    def learn(self, total_timesteps: int) -> Self:
        obs, _ = self.env.reset()
        episode_starts = np.ones(self.num_envs, dtype=bool)

        while self.steps_performed < total_timesteps:
            info: InfoDict = {}

            self.policy.set_train_mode(False)
            with torch.no_grad():
                obs, episode_starts = self.perform_rollout(
                    max_steps=total_timesteps - self.steps_performed,
                    obs=obs,
                    episode_starts=episode_starts,
                    info=info
                )
            self.callback.on_rollout_done(self, self.steps_performed, info)

            if self._should_optimize():
                self.policy.set_train_mode(True)
                self.optimize(obs, episode_starts, info)

                self.callback.on_optimization_done(self, self.steps_performed, info)

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

    def save(self, folder_location: str, name: str, **meta_data):
        os.makedirs(folder_location, exist_ok=True)

        torch.save(self.policy.state_dict(), os.path.join(folder_location, name + POLICY_FILE_SUFFIX))
        joblib.dump(self.buffer, os.path.join(folder_location, name + BUFFER_FILE_SUFFIX))

        with open(os.path.join(folder_location, name + META_DATA_FILE_SUFFIX), 'w') as f:
            json.dump({
                'steps_performed': self.steps_performed,
                **meta_data
            }, f)

    def load(self, folder_location: str, name: str) -> dict[str, Any]:
        os.makedirs(folder_location, exist_ok=True)

        self.policy.load_state_dict(torch.load(os.path.join(folder_location, name + POLICY_FILE_SUFFIX)))
        self.buffer = joblib.load(os.path.join(folder_location, name + BUFFER_FILE_SUFFIX))

        with open(os.path.join(folder_location, name + META_DATA_FILE_SUFFIX), 'r') as f:
            meta_data = json.load(f)

        self.steps_performed = meta_data.get('steps_performed') or self.steps_performed
        return meta_data


    @staticmethod
    def chain_parameters(*modules: nn.Module) -> Iterable[torch.Tensor]:
        return itertools.chain(*[
            module.parameters() for module in modules
        ])
