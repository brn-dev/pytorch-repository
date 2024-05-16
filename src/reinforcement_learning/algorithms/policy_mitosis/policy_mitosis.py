from typing import Callable, Iterable

import numpy as np
from gymnasium import Env
from gymnasium.vector import VectorEnv

from src.model_db.model_db import ModelDB
from src.reinforcement_learning.algorithms.policy_mitosis.policy_mitosis_base import PolicyMitosisBase, \
    PolicyWithEnvAndInfo, TrainPolicyFunction
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policies.policy_initialization import init_policy_using_source
from src.reinforcement_learning.core.policy_info import PolicyInfo
from src.reinforcement_learning.gym.envs.env_wrapping import wrap_env_using_source


class PolicyMitosis(PolicyMitosisBase):

    def __init__(
            self,
            policy_db: ModelDB[PolicyInfo],
            train_policy_function: TrainPolicyFunction,
            env: Env | Callable[[], Env],
            new_init_policy_function: Callable[[], BasePolicy],
            new_wrap_env_function: Callable[[Env | VectorEnv], Env | VectorEnv],
            select_policy_selection_probs: Callable[[Iterable[PolicyInfo]], np.ndarray],
            min_base_ancestors: int,
            rng_seed: int | None,
    ):
        super().__init__(
            policy_db=policy_db,
            train_policy_function=train_policy_function,
            new_init_policy_function=new_init_policy_function,
            new_wrap_env_function=new_wrap_env_function,
            select_policy_selection_probs=select_policy_selection_probs,
            min_base_ancestors=min_base_ancestors,
            rng_seed=rng_seed,
        )

        if isinstance(env, Env):
            self.env = env
        else:
            self.env = env()

    def train_with_mitosis(self, nr_iterations: int):
        for i_iteration in range(nr_iterations):
            policy_info = self.pick_policy_info()

            policy_with_env_and_info = self.create_policy_with_env_and_info(policy_info)

            PolicyMitosisBase.train_policy_iteration(self.train_policy_function, policy_with_env_and_info)

            self.save_policy(policy_with_env_and_info['policy'], policy_with_env_and_info['policy_info'])

    def train_lineage(self, start_policy_policy_id: str, nr_iterations: int):
        policy_id = start_policy_policy_id
        for i_iteration in range(nr_iterations):
            parent_policy_info = self.policy_db.fetch_entry(policy_id)['model_info']
            policy_info = self.create_child_policy_info(parent_policy_info)

            policy_with_env_and_info = self.create_policy_with_env_and_info(policy_info)

            PolicyMitosisBase.train_policy_iteration(self.train_policy_function, policy_with_env_and_info)
            self.save_policy(policy_with_env_and_info['policy'], policy_with_env_and_info['policy_info'])

            policy_id = policy_info['policy_id']

    def create_policy_with_env_and_info(self, policy_info: PolicyInfo) -> PolicyWithEnvAndInfo:
        policy = init_policy_using_source(policy_info['init_policy_source_code'])

        if policy_info['parent_policy_id'] is not None:
            self.policy_db.load_model_state_dict(policy, policy_info['parent_policy_id'])

        return {
            'policy': policy,
            'env': wrap_env_using_source(self.env, policy_info['wrap_env_source_code']),
            'policy_info': policy_info,
        }

    def save_policy(self, policy: BasePolicy, policy_info: PolicyInfo):
        self.policy_db.save_model_state_dict(
            model=policy,
            model_id=policy_info['policy_id'],
            parent_model_id=policy_info['parent_policy_id'],
            model_info=policy_info
        )
