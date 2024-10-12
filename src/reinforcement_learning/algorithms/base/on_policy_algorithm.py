import abc
from typing import TypeVar

import gymnasium
import numpy as np
import torch
from torch import optim

from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.algorithms.base.base_algorithm import BaseAlgorithm, Policy, StashConf, PolicyProvider
from src.reinforcement_learning.core.buffers.rollout.base_rollout_buffer import BaseRolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.infos import InfoDict, stack_infos
from src.reinforcement_learning.core.type_aliases import OptimizerProvider
from src.torch_device import TorchDevice, optimizer_to_device
from src.void_list import VoidList

RolloutBuf = TypeVar('RolloutBuf', bound=BaseRolloutBuffer)


class OnPolicyAlgorithm(BaseAlgorithm[Policy, RolloutBuf, StashConf], abc.ABC):

    def __init__(
            self,
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            policy_optimizer: optim.Optimizer | OptimizerProvider,
            buffer: RolloutBuf,
            gamma: float,
            gae_lambda: float,
            sde_noise_sample_freq: int | None,
            callback: Callback,
            stash_config: StashConf,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype
    ):
        super().__init__(
            env=env,
            policy=policy,
            buffer=buffer,
            gamma=gamma,
            sde_noise_sample_freq=sde_noise_sample_freq,
            callback=callback,
            stash_config=stash_config,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
        )
        self.gae_lambda = gae_lambda

        if isinstance(policy_optimizer, optim.Optimizer):
            self.policy_optimizer = optimizer_to_device(policy_optimizer, self.torch_device)
        else:
            self.policy_optimizer = optimizer_to_device(policy_optimizer(self.policy.parameters()), self.torch_device)

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'gae_lambda': self.gae_lambda,
            'policy_optimizer': self.get_optimizer_hps(self.policy_optimizer),
        })

    def rollout_step(
            self,
            obs: np.ndarray,
            episode_starts: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        action_selector, value_estimates = self.policy(
            torch.tensor(obs, device=self.torch_device, dtype=self.torch_dtype)
        )
        actions = action_selector.get_actions()
        next_obs, rewards, terminated, truncated, info = self.env.step(actions.detach().cpu().numpy())

        if self.stash_config.stash_rollout_action_stds:
            info['action_stds'] = action_selector.distribution.stddev

        self.buffer.add(
            observations=obs,
            rewards=rewards,
            episode_starts=episode_starts,
            actions=actions,
            action_log_probs=action_selector.log_prob(actions),
            value_estimates=value_estimates
        )

        return next_obs, np.logical_or(terminated, truncated), info

    def perform_rollout(
            self,
            max_steps: int,
            obs: np.ndarray,
            episode_starts: np.ndarray,
            info: InfoDict
    ) -> tuple[np.ndarray, np.ndarray]:
        self.buffer.reset()
        self.policy.reset_sde_noise(self.num_envs)

        infos: list[InfoDict] = [] if self.stash_config.stash_rollout_infos else VoidList()
        for _ in range(min(self.buffer.step_size, max_steps)):
            if self.sde_noise_sample_freq is not None and self.steps_performed % self.sde_noise_sample_freq == 0:
                self.policy.reset_sde_noise(self.num_envs)

            obs, episode_starts, step_info = self.rollout_step(obs, episode_starts)
            infos.append(step_info)

            self.steps_performed += 1

        if self.stash_config.stash_rollout_infos:
            info['rollout'] = stack_infos(infos)

        if self.stash_config.stash_last_obs:
            info['last_obs'] = obs
            info['last_episode_starts'] = episode_starts

        return obs, episode_starts
