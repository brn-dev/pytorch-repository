import abc
from typing import Callable, Any, TypeVar, Generic

import gymnasium
import numpy as np
import torch
from gymnasium.vector import VectorEnv

from src.reinforcement_learning.core.buffers.basic_rollout_buffer import BasicRolloutBuffer
from src.reinforcement_learning.core.normalization import NormalizationType, normalize_np_array
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.singleton_vector_env import SingletonVectorEnv


Buffer = TypeVar('Buffer', bound=BasicRolloutBuffer)


class EpisodicRLBase(abc.ABC):


    def __init__(
            self,
            env: gymnasium.Env,
            policy: BasePolicy,
            select_action: Callable[[torch.tensor], tuple[Any, torch.Tensor]],
            buffer: Buffer,
            gamma: float,
            gae_lambda: float,
            normalize_advantages: NormalizationType,
            on_rollout_done: 'RolloutDoneCallback',
            on_optimization_done: 'OptimizationDoneCallback',
    ):
        self.env = self.as_vec_env(env)
        self.policy = policy
        self.select_action = select_action
        self.buffer = buffer

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

        self.on_rollout_done = on_rollout_done
        self.on_optimization_done = on_optimization_done

    @abc.abstractmethod
    def optimize(self, last_obs: np.ndarray, last_dones: np.ndarray) -> None:
        raise NotImplemented

    def rollout_step(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        action_preds, extra_predictions = self.policy.process_obs(state)
        actions, action_log_probs = self.select_action(action_preds)

        next_states, rewards, terminated, truncated, info = self.env.step(actions)

        self.buffer.add(
            observations=state,
            rewards=rewards,
            episode_starts=np.logical_or(terminated, truncated),
            action_log_probs=action_log_probs,
            **extra_predictions
        )

        return next_states, rewards, terminated, truncated, info


    def perform_rollout(self, max_steps: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        obs, info = self.env.reset()

        terminated = np.empty((self.env.num_envs,), dtype=bool)
        truncated = np.empty((self.env.num_envs,), dtype=bool)
        step = 0
        for step in range(min(self.buffer.buffer_size, max_steps)):
            obs, rewards, terminated, truncated, info = self.rollout_step(obs)

        return step, obs, terminated, truncated


    def train(self, num_steps: int):

        step = 0
        while step < num_steps:

            steps_performed, last_obs, last_terminated, last_truncated = self.perform_rollout(num_steps - step)
            step += steps_performed

            self.on_rollout_done(self, step, last_obs, last_terminated, last_truncated)

            self.optimize(last_obs, np.logical_or(last_terminated, last_truncated))

            self.on_optimization_done(self, step)

            self.buffer.reset()

    # Adapted from
    # https://github.com/DLR-RM/stable-baselines3/blob/5623d98f9d6bcfd2ab450e850c3f7b090aef5642/stable_baselines3/common/buffers.py#L402
    def compute_gae_and_returns(
            self,
            last_values: torch.Tensor,
            last_dones: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        last_values = last_values.clone().cpu().numpy()

        value_estimates = torch.stack(self.buffer.value_estimates).squeeze(-1).detach().cpu().numpy()

        advantages = np.zeros_like(self.buffer.rewards)

        gae = 0
        for step in reversed(range(self.buffer.buffer_size)):
            if step == self.buffer.buffer_size - 1:
                next_non_terminal = 1.0 - last_dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.buffer.episode_starts[step + 1]
                next_values = value_estimates[step + 1]
            delta = self.buffer.rewards[step] + self.gamma * next_values * next_non_terminal - value_estimates[step]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae

            advantages[step] = gae

        returns = advantages + value_estimates

        if self.normalize_advantages is not None:
            advantages = normalize_np_array(advantages, normalization_type=self.normalize_advantages)

        return advantages, returns

    @staticmethod
    def as_vec_env(env: gymnasium.Env):
        return env if isinstance(env, VectorEnv) else SingletonVectorEnv(env)


EpisodicRLBaseDerived = TypeVar('EpisodicRLBaseDerived', bound=EpisodicRLBase)
RolloutDoneCallback = Callable[[EpisodicRLBaseDerived, int, np.ndarray, np.ndarray, np.ndarray], None]
OptimizationDoneCallback = Callable[[EpisodicRLBaseDerived, int], None]
