import inspect
from typing import Any, Callable, Optional, TypedDict, Iterable

import numpy as np
from gymnasium import Env
from gymnasium.vector import VectorEnv

from src.datetime import get_current_timestamp
from src.model_db.model_db import ModelDB, ModelEntry
from src.np_functions import softmax
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policies.policy_initialization import init_policy_using_source
from src.reinforcement_learning.core.policy_info import PolicyInfo
from src.reinforcement_learning.gym.envs.env_wrapping import wrap_env_using_source

ALPHANUMERIC_ALPHABET = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')


class PolicyWithEnvAndInfo(TypedDict):
    policy: BasePolicy
    env: Env

    policy_id: str
    parent_policy_id: str | None
    policy_info: PolicyInfo

class PolicyMitosis:

    def __init__(
            self,
            policy_db: ModelDB[PolicyInfo],
            policy_train_function: Callable[[PolicyWithEnvAndInfo], tuple[int, float]],
            env: VectorEnv,
            new_init_policy_function: Callable[[], BasePolicy],
            new_wrap_env_function: Callable[[Env | VectorEnv], Env | VectorEnv],
            select_policy_selection_scores: Callable[[Iterable[PolicyInfo]], np.ndarray],
            policy_selection_temperature: float,
            min_base_ancestors: int,
            # max_active_lineages: int | None, TODO
            rng_seed: int | None,
            _globals: dict[str, Any],
    ):
        self.policy_db = policy_db
        self.policy_train_function = policy_train_function
        self.env = env

        self.new_init_policy_function = new_init_policy_function
        self.new_init_policy_source_code = inspect.getsource(new_init_policy_function)

        self.new_wrap_env_function = new_wrap_env_function
        self.new_wrap_env_source_code = inspect.getsource(new_wrap_env_function)

        self.select_policy_selection_scores = select_policy_selection_scores
        self.policy_selection_temperature = policy_selection_temperature
        self.min_base_ancestors = min_base_ancestors

        self.rng = np.random.default_rng(rng_seed)
        self._globals = _globals

        self.policy_id_random_alphanumeric_length = 6
        self.sufficient_base_ancestors = False

    def train(self, max_iterations: int):
        for i_iteration in range(max_iterations):
            policy_with_env_and_info = self.pick_policy()

            steps_trained, score = self.policy_train_function(policy_with_env_and_info)

            policy_info = policy_with_env_and_info['policy_info']
            policy_info['steps_trained'] += steps_trained
            policy_info['score'] = score

            self.policy_db.save_model_state_dict(
                model=policy_with_env_and_info['policy'],
                model_id=policy_with_env_and_info['policy_id'],
                parent_model_id=policy_with_env_and_info['parent_policy_id'],
                model_info=policy_info
            )

    def pick_policy(self) -> PolicyWithEnvAndInfo:
        nr_policies = len(self.policy_db)
        sufficient_base_ancestors = self.eval_sufficient_base_ancestors()

        if not sufficient_base_ancestors or self.rng.random() < 1.0 / (nr_policies + 1):
            return self.create_new_policy()
        else:
            return self.create_child_policy()

    def create_new_policy(self) -> PolicyWithEnvAndInfo:
        policy_id = self.create_policy_id()

        policy = self.new_init_policy_function()
        wrapped_env = self.new_wrap_env_function(self.env)

        policy_info = {
            'score': -1e6,
            'steps_trained': 0,
            'init_policy_source_code': self.new_init_policy_source_code,
            'wrap_env_source_code': self.new_wrap_env_source_code,
        }

        return {
            'policy': policy,
            'env': wrapped_env,
            'policy_id': policy_id,
            'parent_policy_id': None,
            'policy_info': policy_info
        }


    def create_child_policy(self) -> PolicyWithEnvAndInfo:
        policy_id = self.create_policy_id()

        policy_entries = self.policy_db.all_entries()
        nr_policies = len(policy_entries)

        policy_scores = self.select_policy_selection_scores(entry['model_info'] for entry in policy_entries)
        policy_probs = softmax(policy_scores, temperature=self.policy_selection_temperature)

        print('policy selection probs = \n\t' + '\n\t'.join(
            f'{policy_entries[i]["model_id"]}: {p = :8.6f}, '
            f'selection_score = {policy_scores[i]:6.3f}, '
            f'original_score = {policy_entries[i]["model_info"]["score"]:7.3f}, '
            f'steps = {policy_entries[i]["model_info"]["steps_trained"]}'
            for i, p
            in enumerate(policy_probs)
        ))

        selected_parent_policy_index = self.rng.choice(range(nr_policies), p=policy_probs)

        parent_policy_entry: ModelEntry[PolicyInfo] = policy_entries[selected_parent_policy_index]

        policy_info = parent_policy_entry['model_info'].copy()
        init_policy_source = policy_info['init_policy_source_code']
        wrap_env_source = policy_info['wrap_env_source_code']

        policy = init_policy_using_source(init_policy_source, self._globals)
        wrapped_env = wrap_env_using_source(self.env, wrap_env_source, self._globals)

        self.policy_db.load_model_state_dict(policy, parent_policy_entry['model_id'])

        parent_policy_id = parent_policy_entry['model_id']

        return {
            'policy': policy,
            'env': wrapped_env,
            'policy_id': policy_id,
            'parent_policy_id': parent_policy_id,
            'policy_info': policy_info
        }

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

