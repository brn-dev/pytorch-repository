import abc
from typing import Callable, Any, TypeVar

import gymnasium
import numpy as np
from gymnasium.vector import VectorEnv
from torch import optim

from src.reinforcement_learning.core.buffers.basic_rollout_buffer import BasicRolloutBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.infos import InfoDict, stack_infos
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.gym.envs.singleton_vector_env import SingletonVectorEnv

Buffer = TypeVar('Buffer', bound=BasicRolloutBuffer)

Policy = TypeVar('Policy', bound=BasePolicy)
PolicyProvider = Callable[[], Policy]


class RLBase(abc.ABC):


    def __init__(
            self,
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            policy_optimizer: optim.Optimizer | Callable[[BasePolicy], optim.Optimizer],
            buffer: Buffer,
            gamma: float,
            gae_lambda: float,
            reset_env_between_rollouts: bool,
            callback: Callback,
    ):
        self.env = self.as_vec_env(env)
        self.policy = policy if isinstance(policy, BasePolicy) else policy()
        self.policy_optimizer = (
            policy_optimizer
            if isinstance(policy_optimizer, optim.Optimizer)
            else policy_optimizer(policy)
        )
        self.buffer = buffer

        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.reset_env_between_rollouts = reset_env_between_rollouts

        self.callback = callback

    @abc.abstractmethod
    def optimize(
            self,
            last_obs: np.ndarray,
            last_dones: np.ndarray,
            info: InfoDict
    ) -> None:
        raise NotImplemented

    def rollout_step(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        actions_dist, extra_predictions = self.policy.process_obs(obs)
        actions = actions_dist.sample()

        next_states, rewards, terminated, truncated, info = self.env.step(actions.detach().cpu().numpy())

        self.buffer.add(
            observations=obs,
            rewards=rewards,
            episode_starts=np.logical_or(terminated, truncated),
            actions=actions,
            action_log_probs=actions_dist.log_prob(actions),
            **extra_predictions
        )

        return next_states, rewards, terminated, truncated, info


    def perform_rollout(
            self,
            max_steps: int,
            obs: np.ndarray,
            info: InfoDict
    ) -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:

        terminated = np.empty((self.env.num_envs,), dtype=bool)
        truncated = np.empty((self.env.num_envs,), dtype=bool)
        step = 0

        infos: list[InfoDict] = []
        for step in range(min(self.buffer.buffer_size, max_steps)):
            obs, rewards, terminated, truncated, step_info = self.rollout_step(obs)
            infos.append(step_info)

        info['rollout'] = stack_infos(infos)

        info['rollout_last_obs'] = obs
        info['rollout_last_terminated'] = terminated
        info['rollout_last_truncated'] = truncated

        return step + 1, obs, terminated, truncated


    def train(self, num_steps: int):
        obs: np.ndarray = np.empty(())

        step = 0
        while step < num_steps:
            info: InfoDict = {}

            if step == 0 or self.reset_env_between_rollouts:
                obs, reset_info = self.env.reset()
                info['reset'] = reset_info
                print('envs reset')

            steps_performed, obs, last_terminated, last_truncated = self.perform_rollout(
                max_steps=num_steps - step,
                obs=obs,
                info=info
            )
            step += steps_performed

            self.callback.on_rollout_done(self, step, info)

            self.optimize(obs, np.logical_or(last_terminated, last_truncated), info)

            self.callback.on_optimization_done(self, step, info)

            self.buffer.reset()

    @staticmethod
    def is_vec_env(env: gymnasium.Env):
        try:
            env.get_wrapper_attr('num_envs')
            return True
        except AttributeError:
            return False

    @staticmethod
    def as_vec_env(env: gymnasium.Env):
        if not RLBase.is_vec_env(env):
            return SingletonVectorEnv(env)

        return env

