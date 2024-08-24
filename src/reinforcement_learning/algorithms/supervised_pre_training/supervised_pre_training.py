from typing import Callable, Self, TypeVar, Type

import gymnasium
import numpy as np
import torch
from torch import optim

from src.reinforcement_learning.algorithms.base.on_policy_algorithm import OnPolicyAlgorithm, Policy, \
    PolicyProvider, LoggingConfig
from src.reinforcement_learning.core.buffers.rollout.rollout_buffer import BasicRolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.infos import InfoDict
from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy
from src.reinforcement_learning.gym.singleton_vector_env import as_vec_env
from src.torch_device import TorchDevice

Buffer = TypeVar('Buffer', bound=BasicRolloutBuffer)
BufferType = Type[Buffer]


class SupervisedPreTraining(OnPolicyAlgorithm):

    def __init__(
            self,
            compute_supervised_objective: Callable[[Buffer, np.ndarray, np.ndarray, InfoDict], torch.Tensor],
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            policy_optimizer: optim.Optimizer | Callable[[ActorCriticPolicy], optim.Optimizer],
            buffer_size: int,
            buffer_type: BufferType = BasicRolloutBuffer,
            gamma: float = 1.0,
            gae_lambda: float = 1.0,
            sde_noise_sample_freq: int | None = None,
            reset_env_between_rollouts: bool = False,
            callback: Callback[Self] = None,
            logging_config: LoggingConfig = None,
            torch_device: TorchDevice = 'cpu',
    ):
        env, num_envs = as_vec_env(env)

        super().__init__(
            env=env,
            policy=policy,
            policy_optimizer=policy_optimizer,
            buffer=buffer_type(buffer_size, num_envs, env.observation_space.shape),
            gamma=gamma,
            gae_lambda=gae_lambda,
            sde_noise_sample_freq=sde_noise_sample_freq,
            reset_env_between_rollouts=reset_env_between_rollouts,
            callback=callback or Callback(),
            logging_config=logging_config or LoggingConfig(),
            torch_device=torch_device,
        )

        self.compute_supervised_objective = compute_supervised_objective

    def optimize(
            self,
            last_obs: np.ndarray,
            last_episode_starts: np.ndarray,
            info: InfoDict,
    ) -> None:
        self.policy_optimizer.zero_grad()
        objective = self.compute_supervised_objective(self.buffer, last_obs, last_episode_starts, info)
        objective.backward()
        self.policy_optimizer.step()



