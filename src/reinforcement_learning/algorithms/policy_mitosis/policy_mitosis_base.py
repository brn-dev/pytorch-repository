import abc
import inspect
from typing import Any, Callable, TypedDict, Iterable

import numpy as np
from gymnasium import Env
from gymnasium.vector import VectorEnv

from src.datetime import get_current_timestamp
from src.model_db.model_db import ModelDB, ModelEntry
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policies.policy_initialization import init_policy_using_source
from src.reinforcement_learning.core.policy_info import PolicyInfo
from src.reinforcement_learning.gym.envs.env_wrapping import wrap_env_using_source

ALPHANUMERIC_ALPHABET = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')


class PolicyWithEnvAndInfo(TypedDict):
    policy: BasePolicy
    env: Env
    policy_info: PolicyInfo


TrainPolicyFunction = Callable[[PolicyWithEnvAndInfo], tuple[int, float]]


class PolicyMitosisBase(abc.ABC):

    def __init__(
            self,
            policy_db: ModelDB[PolicyInfo],
            train_policy_function: TrainPolicyFunction,
            new_init_policy_function: Callable[[], BasePolicy],
            new_wrap_env_function: Callable[[Env | VectorEnv], Env | VectorEnv],
            select_policy_selection_probs: Callable[[Iterable[PolicyInfo]], np.ndarray],
            min_base_ancestors: int,
            rng_seed: int | None,
    ):
        self.policy_db = policy_db
        self.train_policy_function = train_policy_function

        self.new_init_policy_function = new_init_policy_function
        self.new_init_policy_source_code = inspect.getsource(new_init_policy_function)

        self.new_wrap_env_function = new_wrap_env_function
        self.new_wrap_env_source_code = inspect.getsource(new_wrap_env_function)

        self.select_policy_selection_probs = select_policy_selection_probs
        self.min_base_ancestors = min_base_ancestors

        self.rng = np.random.default_rng(rng_seed)

        self.policy_id_random_alphanumeric_length = 6
        self.sufficient_base_ancestors = False

    @abc.abstractmethod
    def train_with_mitosis(self, nr_iterations: int):
        raise NotImplementedError

    @staticmethod
    def train_policy_iteration(
            train_policy_function: TrainPolicyFunction,
            policy_with_env_and_info: PolicyWithEnvAndInfo
    ):
        steps_trained, score = train_policy_function(policy_with_env_and_info)

        policy_info = policy_with_env_and_info['policy_info']
        policy_info['steps_trained'] += steps_trained
        policy_info['score'] = score

    def pick_policy_info(self) -> PolicyInfo:
        nr_policies = len(self.policy_db)
        sufficient_base_ancestors = self.eval_sufficient_base_ancestors()

        if not sufficient_base_ancestors or self.rng.random() < 1.0 / (nr_policies + 1):
            return self.create_new_policy_info()
        else:
            selected_parent_policy_info = self.select_parent_policy_info()
            return self.create_child_policy_info(selected_parent_policy_info)

    def create_new_policy_info(self) -> PolicyInfo:
        return {
            'policy_id': self.create_policy_id(),
            'parent_policy_id': None,
            'score': -1e6,
            'steps_trained': 0,
            'init_policy_source_code': self.new_init_policy_source_code,
            'wrap_env_source_code': self.new_wrap_env_source_code,
        }

    def select_parent_policy_info(self) -> PolicyInfo:
        policy_entries = self.policy_db.all_entries()
        nr_policies = len(policy_entries)

        policy_probs = self.select_policy_selection_probs(entry['model_info'] for entry in policy_entries)

        print('policy selection probs = \n\t' + '\n\t'.join(
            f'{policy_entries[i]["model_id"]}: {p = :8.6f}, '
            f'scores = {policy_entries[i]["model_info"]["score"]:7.3f}, '
            f'steps = {policy_entries[i]["model_info"]["steps_trained"]}'
            for i, p
            in enumerate(policy_probs)
        ))

        selected_parent_policy_index = self.rng.choice(range(nr_policies), p=policy_probs)
        selected_parent_policy: ModelEntry[PolicyInfo] = policy_entries[selected_parent_policy_index]
        return selected_parent_policy['model_info']

    def create_child_policy_info(self, parent_policy_info: PolicyInfo) -> PolicyInfo:
        policy_info: PolicyInfo = parent_policy_info.copy()

        policy_info['parent_policy_id'] = policy_info['policy_id']
        policy_info['policy_id'] = self.create_policy_id()

        return policy_info

    def save_policy(self, policy: BasePolicy, policy_info: PolicyInfo):
        self.policy_db.save_model_state_dict(
            model=policy,
            model_id=policy_info['policy_id'],
            parent_model_id=policy_info['parent_policy_id'],
            model_info=policy_info
        )

    def create_policy_id(self) -> str:
        random_alphanumeric = ''.join(np.random.choice(
            ALPHANUMERIC_ALPHABET,
            self.policy_id_random_alphanumeric_length
        ))
        return f'{get_current_timestamp()}~{random_alphanumeric}'

    def eval_sufficient_base_ancestors(self):
        if self.sufficient_base_ancestors:
            return True

        self.sufficient_base_ancestors = len([
            policy_entry
            for policy_entry
            in self.policy_db.all_entries()
            if policy_entry['parent_model_id'] is None
        ]) >= self.min_base_ancestors

        return self.sufficient_base_ancestors

