import abc
from typing import Callable, Any, TypeVar

import gymnasium
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch import optim

from src.reinforcement_learning.core.buffers.basic_rollout_buffer import BasicRolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.normalization import NormalizationType
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.gym.envs.singleton_vector_env import SingletonVectorEnv

Buffer = TypeVar('Buffer', bound=BasicRolloutBuffer)


class EpisodicRLBase(abc.ABC):


    def __init__(
            self,
            env: gymnasium.Env,
            policy: BasePolicy,
            policy_optimizer: optim.Optimizer | Callable[[BasePolicy], optim.Optimizer],
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            buffer: Buffer,
            gamma: float,
            gae_lambda: float,
            normalize_advantages: NormalizationType | None,
            callback: Callback,
    ):
        self.env = self.as_vec_env(env)
        self.policy = policy
        self.policy_optimizer = (
            policy_optimizer
            if isinstance(policy_optimizer, optim.Optimizer)
            else policy_optimizer(policy)
        )
        self.select_action = select_action
        self.buffer = buffer

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

        self.callback = callback

    @abc.abstractmethod
    def compute_objectives(
            self,
            last_obs: np.ndarray,
            last_dones: np.ndarray,
            info: dict[str, Any],
    ) -> list[torch.Tensor]:
        raise NotImplemented

    def optimize(
            self,
            last_obs: np.ndarray,
            last_dones: np.ndarray,
            info: dict[str, Any]
    ) -> None:
        objectives = self.compute_objectives(last_obs, last_dones, info)

        objective = torch.stack(objectives).sum()

        info['objective'] = objective

        self.policy_optimizer.zero_grad()
        objective.backward()
        self.policy_optimizer.step()

    def rollout_step(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        action_logits, extra_predictions = self.policy.process_obs(obs)
        actions, action_log_probs = self.select_action(action_logits)

        next_states, rewards, terminated, truncated, info = self.env.step(actions)

        self.buffer.add(
            observations=obs,
            rewards=rewards,
            episode_starts=np.logical_or(terminated, truncated),
            action_log_probs=action_log_probs,
            **extra_predictions
        )

        return next_states, rewards, terminated, truncated, info


    def perform_rollout(self, max_steps: int, info: dict[str, Any]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        obs, _ = self.env.reset()

        terminated = np.empty((self.env.num_envs,), dtype=bool)
        truncated = np.empty((self.env.num_envs,), dtype=bool)
        step = 0
        for step in range(min(self.buffer.buffer_size, max_steps)):
            obs, rewards, terminated, truncated, _ = self.rollout_step(obs)

        info['last_obs'] = obs
        info['last_terminated'] = terminated
        info['last_truncated'] = truncated

        return step + 1, obs, terminated, truncated


    def train(self, num_steps: int):
        # TODO: flag for not resetting env after resetting buffer

        step = 0
        while step < num_steps:
            info: dict[str, Any] = {}

            steps_performed, last_obs, last_terminated, last_truncated = self.perform_rollout(num_steps - step, info)
            step += steps_performed

            self.callback.on_rollout_done(self, step, info)

            self.optimize(last_obs, np.logical_or(last_terminated, last_truncated), info)

            self.callback.on_optimization_done(self, step, info)

            self.buffer.reset()

    @staticmethod
    def as_vec_env(env: gymnasium.Env):
        return env if isinstance(env, VectorEnv) else SingletonVectorEnv(env)
