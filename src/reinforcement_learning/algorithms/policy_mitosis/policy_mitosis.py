from typing import Callable, Iterable

import numpy as np
from gymnasium import Env
from torch import optim

from src.model_db.model_db import ModelDB
from src.reinforcement_learning.algorithms.policy_mitosis.mitosis_policy_info import MitosisPolicyInfo
from src.reinforcement_learning.algorithms.policy_mitosis.policy_mitosis_base import PolicyMitosisBase, \
    TrainInfo, TrainPolicyFunction
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policy_construction import PolicyConstruction, PolicyInitializationInfo


class PolicyMitosis(PolicyMitosisBase):

    def __init__(
            self,
            policy_db: ModelDB[MitosisPolicyInfo],
            train_policy_function: TrainPolicyFunction,
            env: Env | Callable[[], Env],
            new_policy_initialization_info: PolicyInitializationInfo,
            new_policy_prob_function: Callable[[int, int], float],
            select_policy_selection_probs: Callable[[Iterable[MitosisPolicyInfo]], np.ndarray],
            min_primordial_ancestors: int,
            save_optimizer_state_dicts: bool,
            load_optimizer_state_dicts: bool,
            rng_seed: int | None,
    ):
        super().__init__(
            policy_db=policy_db,
            train_policy_function=train_policy_function,
            new_policy_initialization_info=new_policy_initialization_info,
            new_policy_prob_function=new_policy_prob_function,
            select_policy_selection_probs=select_policy_selection_probs,
            min_primordial_ancestors=min_primordial_ancestors,
            save_optimizer_state_dicts=save_optimizer_state_dicts,
            load_optimizer_state_dicts=load_optimizer_state_dicts,
            rng_seed=rng_seed,
        )

        if isinstance(env, Env):
            self.env = env
        else:
            self.env = env()

    def train_with_mitosis(self, nr_iterations: int):
        for i_iteration in range(nr_iterations):
            policy_info = self.pick_policy_info()

            train_info = self.create_train_info(policy_info)

            PolicyMitosisBase.train_policy_iteration(self.train_policy_function, train_info)

            self.save_policy(train_info['policy'], train_info['policy_info'])

    def train_lineage(self, start_policy_policy_id: str, nr_iterations: int):
        policy_id = start_policy_policy_id
        for i_iteration in range(nr_iterations):
            parent_policy_info = self.policy_db.fetch_entry(policy_id)['model_info']
            policy_info = self.create_child_policy_info(parent_policy_info)

            train_info = self.create_train_info(policy_info)

            PolicyMitosisBase.train_policy_iteration(self.train_policy_function, train_info)
            self.save_policy(train_info['policy'], train_info['optimizer'], train_info['policy_info'])

            policy_id = policy_info['policy_id']

    def create_train_info(self, policy_info: MitosisPolicyInfo) -> TrainInfo:
        policy, optimizer, env = PolicyConstruction.init_from_info(policy_info['initialization_info'], self.env)

        if policy_info['parent_policy_id'] is not None:
            self.policy_db.load_model_state_dict(
                policy_info['parent_policy_id'],
                policy,
                optimizer if self.load_optimizer_state_dicts else None
            )

        return {
            'policy': policy,
            'optimizer': optimizer,
            'env': env,
            'policy_info': policy_info,
        }

    def save_policy(self, policy: BasePolicy, optimizer: optim.Optimizer, policy_info: MitosisPolicyInfo):
        self.policy_db.save_model_state_dict(
            model_id=policy_info['policy_id'],
            parent_model_id=policy_info['parent_policy_id'],
            model_info=policy_info,
            model=policy,
            optimizer=optimizer if self.save_optimizer_state_dicts else None,
        )
