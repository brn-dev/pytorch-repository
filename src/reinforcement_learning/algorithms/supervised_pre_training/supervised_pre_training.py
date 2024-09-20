from typing import Callable, Self, TypeVar, Type

import gymnasium
import numpy as np
import torch
from torch import optim

from src.reinforcement_learning.core.logging import LoggingConfig
from src.reinforcement_learning.algorithms.base.on_policy_algorithm import OnPolicyAlgorithm, Policy, \
    PolicyProvider
from src.reinforcement_learning.core.buffers.rollout.rollout_buffer import RolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.infos import InfoDict
from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy
from src.reinforcement_learning.gym.singleton_vector_env import as_vec_env
from src.torch_device import TorchDevice

Buffer = TypeVar('Buffer', bound=RolloutBuffer)
BufferType = Type[Buffer]


class SupervisedPreTraining(OnPolicyAlgorithm):

    def __init__(
            self,
            compute_supervised_loss: Callable[[Buffer, np.ndarray, np.ndarray, InfoDict], torch.Tensor],
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            policy_optimizer: optim.Optimizer | Callable[[ActorCriticPolicy], optim.Optimizer],
            buffer_size: int,
            buffer_type: BufferType = RolloutBuffer,
            gamma: float = 1.0,
            gae_lambda: float = 1.0,
            sde_noise_sample_freq: int | None = None,
            callback: Callback[Self] = None,
            logging_config: LoggingConfig = None,
            torch_device: TorchDevice = 'cpu',
            torch_dtype: torch.dtype = torch.float32,
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
            callback=callback or Callback(),
            logging_config=logging_config or LoggingConfig(),
            torch_device=torch_device,
            torch_dtype=torch_dtype,
        )

        self.compute_supervised_loss = compute_supervised_loss

    def optimize(
            self,
            last_obs: np.ndarray,
            last_episode_starts: np.ndarray,
            info: InfoDict,
    ) -> None:
        self.policy_optimizer.zero_grad()
        loss = self.compute_supervised_loss(self.buffer, last_obs, last_episode_starts, info)
        loss.backward()
        self.policy_optimizer.step()



