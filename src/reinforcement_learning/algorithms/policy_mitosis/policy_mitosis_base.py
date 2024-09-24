import abc
from typing import Callable, TypedDict, Iterable, Any

import numpy as np
from gymnasium import Env
from torch import optim

from src.datetime import get_current_timestamp
from src.id_generation import generate_timestamp_id
from src.model_db.model_db import ModelDB, ModelEntry
from src.reinforcement_learning.algorithms.policy_mitosis.mitosis_policy_info import MitosisPolicyInfo
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policy_construction import PolicyInitializationInfo, PolicyConstructionOverride

ALPHANUMERIC_ALPHABET = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')


class TrainInfo(TypedDict):
    policy: BasePolicy
    optimizer: optim.Optimizer
    env: Env
    policy_info: MitosisPolicyInfo


class TrainResultInfo(TypedDict):
    steps_trained: int
    optimizations_done: int
    score: float
    extra_infos: dict[str, Any]


TrainPolicyFunction = Callable[[TrainInfo], TrainResultInfo]


class PolicyMitosisBase(abc.ABC):

    def __init__(
            self,
            policy_db: ModelDB[MitosisPolicyInfo],
            train_policy_function: TrainPolicyFunction,
            new_policy_initialization_info: PolicyInitializationInfo,
            new_policy_prob_function: Callable[[int, int], float],
            select_policy_selection_probs: Callable[[Iterable[MitosisPolicyInfo]], np.ndarray],
            min_primordial_ancestors: int,
            save_optimizer_state_dicts: bool,
            load_optimizer_state_dicts: bool,
            rng_seed: int | None,
    ):
        self.policy_db = policy_db
        self.train_policy_function = train_policy_function

        self.new_policy_initialization_info = new_policy_initialization_info
        self.new_policy_prob_function = new_policy_prob_function

        self.select_policy_selection_probs = select_policy_selection_probs
        self.min_primordial_ancestors = min_primordial_ancestors

        self.save_optimizer_state_dicts = save_optimizer_state_dicts
        self.load_optimizer_state_dicts = load_optimizer_state_dicts

        self.rng = np.random.default_rng(rng_seed)

        self.sufficient_primordial_ancestors = False

    @abc.abstractmethod
    def train_with_mitosis(self, nr_iterations: int):
        raise NotImplementedError

    @staticmethod
    def train_policy_iteration(
            train_policy_function: TrainPolicyFunction,
            train_info: TrainInfo,
    ):
        train_result = train_policy_function(train_info)
        steps_trained = train_result['steps_trained']

        policy_info = train_info['policy_info']

        policy_info['steps_trained'] += steps_trained
        policy_info['optimizations_done'] += train_result['optimizations_done']
        policy_info['score'] = train_result['score']
        policy_info['extra_infos'].update(train_result['extra_infos'])

        try:
            num_envs = train_info['env'].get_wrapper_attr("num_envs")
        except AttributeError:
            num_envs = 1

        policy_info['env_steps_trained'] += steps_trained * num_envs

    def pick_policy_info(self) -> MitosisPolicyInfo:
        nr_policies = len(self.policy_db)
        nr_primordial_ancestors = len([
            policy_entry
            for policy_entry
            in self.policy_db.all_entries()
            if policy_entry['parent_model_id'] is None
        ])

        sufficient_primordial_ancestors = nr_primordial_ancestors >= self.min_primordial_ancestors
        new_policy_prob = self.new_policy_prob_function(nr_policies, nr_primordial_ancestors)

        if not sufficient_primordial_ancestors or self.rng.random() < new_policy_prob:
            return self.create_new_policy_info()
        else:
            selected_parent_policy_info = self.select_parent_policy_info()
            return self.create_child_policy_info(selected_parent_policy_info)

    def create_new_policy_info(self) -> MitosisPolicyInfo:
        return {
            'policy_id': self.create_policy_id(),
            'parent_policy_id': None,
            'score': -1e6,
            'steps_trained': 0,
            'env_steps_trained': 0,
            'optimizations_done': 0,
            'extra_infos': {},
            'initialization_info': self.new_policy_initialization_info,
        }

    def select_parent_policy_info(self) -> MitosisPolicyInfo:
        policy_entries = self.policy_db.all_entries()
        nr_policies = len(policy_entries)

        policy_probs = self.select_policy_selection_probs(entry['model_info'] for entry in policy_entries)

        selected_parent_policy_index = self.rng.choice(range(nr_policies), p=policy_probs)
        selected_parent_policy: ModelEntry[MitosisPolicyInfo] = policy_entries[selected_parent_policy_index]
        return selected_parent_policy['model_info']

    def create_child_policy_info(self, parent_policy_info: MitosisPolicyInfo) -> MitosisPolicyInfo:
        policy_info: MitosisPolicyInfo = parent_policy_info.copy()

        policy_info['parent_policy_id'] = policy_info['policy_id']
        policy_info['policy_id'] = self.create_policy_id()

        return policy_info

    @staticmethod
    def create_policy_id() -> str:
        return generate_timestamp_id()

