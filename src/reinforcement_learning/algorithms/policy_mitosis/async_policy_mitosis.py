from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable

import numpy as np
from gymnasium import Env
from gymnasium.vector import VectorEnv
from overrides import override

from src.model_db.model_db import ModelDB
from src.reinforcement_learning.algorithms.policy_mitosis.policy_mitosis_base import PolicyMitosisBase, \
    TrainPolicyFunction
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policy_info import PolicyInfo
import multiprocessing as mp
import dill


class AsyncPolicyMitosis(PolicyMitosisBase):

    def __init__(
            self,
            policy_db: ModelDB[PolicyInfo],
            train_policy_function: TrainPolicyFunction,
            create_env: Callable[[], Env],
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

        self.create_env = create_env

    def train_with_mitosis(self, nr_iterations: int):
        with ThreadPoolExecutor() as pool:
            futures = [
                pool.submit(self.worker_task, worker_nr, nr_iterations)
                for worker_nr
                in range(self.num_workers)
            ]
            for future in futures:
                future.result()

    def bump_num_iterations_started(self) -> int:
        with self.num_iterations_started_lock:
            self.num_iterations_started += 1
            return self.num_iterations_started

    def worker_task(self, worker_nr: int, nr_iterations: int):
        i_iteration = self.bump_num_iterations_started()
        while i_iteration <= nr_iterations:
            print(f'Worker {worker_nr} starting iteration {i_iteration}')

            policy_with_env_and_info = self.pick_policy()
            self.train_iteration(policy_with_env_and_info)

            print(f'Worker {worker_nr} finished iteration {i_iteration}')

            i_iteration = self.bump_num_iterations_started()


def _worker():
    pass