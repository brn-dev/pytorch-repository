import inspect
from typing import Any, Callable, Optional, TypedDict

import numpy as np
from gymnasium import Env
from gymnasium.vector import VectorEnv

from src.datetime import get_current_timestamp
from src.model_db.model_db import ModelDB, ModelEntry
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policies.policy_initialization import init_policy_using_source
from src.reinforcement_learning.core.policy_info import PolicyInfo
from src.reinforcement_learning.gym.envs.env_wrapping import wrap_env_using_source


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
            _globals: dict[str, Any],
            rng_seed: int | None = None,
    ):
        self.policy_db = policy_db
        self.policy_train_function = policy_train_function
        self.env = env

        self.new_init_policy_function = new_init_policy_function
        self.new_init_policy_source_code = inspect.getsource(new_init_policy_function)

        self.new_wrap_env_function = new_wrap_env_function
        self.new_wrap_env_source_code = inspect.getsource(new_wrap_env_function)

        self.rng = np.random.default_rng(rng_seed)
        self._globals = _globals

    def train(self, max_iterations: int):
        for i_iteration in range(max_iterations):
            policy_with_env_and_info = self.pick_policy()

            steps_trained, score = self.policy_train_function(policy_with_env_and_info)

            policy_info = policy_with_env_and_info['policy_info']
            policy_info['steps_trained'] += steps_trained
            policy_info['score'] += score

            self.policy_db.save_model_state_dict(
                model=policy_with_env_and_info['policy'],
                model_id=policy_with_env_and_info['policy_id'],
                parent_model_id=policy_with_env_and_info['parent_policy_id'],
                model_info=policy_info
            )

    def pick_policy(self) -> PolicyWithEnvAndInfo:
        policy_entries = self.policy_db.all_entries()
        nr_policies = len(policy_entries)

        policy_info: PolicyInfo
        parent_policy_id: Optional[str]
        if self.rng.random() < 1.0 / (nr_policies + 1):
            policy = self.new_init_policy_function()
            wrapped_env = self.new_wrap_env_function(self.env)

            policy_info = {
                'score': 1e-6,
                'steps_trained': 0,
                'init_policy_source_code': self.new_init_policy_source_code,
                'wrap_env_source_code': self.new_wrap_env_source_code,
            }

            parent_policy_id = None
        else:
            policy_scores = np.array([entry['model_info']['score'] for entry in policy_entries])
            policy_scores = policy_scores - policy_scores.min() + 1e-2
            chosen_parent_policy_index = self.rng.choice(range(nr_policies), p=policy_scores / policy_scores.sum())

            parent_policy_entry: ModelEntry[PolicyInfo] = policy_entries[chosen_parent_policy_index]

            policy_info = parent_policy_entry['model_info'].copy()
            init_policy_source = policy_info['init_policy_source_code']
            wrap_env_source = policy_info['wrap_env_source_code']

            policy = init_policy_using_source(init_policy_source, self._globals)
            wrapped_env = wrap_env_using_source(self.env, wrap_env_source, self._globals)

            parent_policy_id = parent_policy_entry['model_id']

        policy_id = get_current_timestamp()

        return {
            'policy': policy,
            'env': wrapped_env,
            'policy_id': policy_id,
            'parent_policy_id': parent_policy_id,
            'policy_info': policy_info
        }
