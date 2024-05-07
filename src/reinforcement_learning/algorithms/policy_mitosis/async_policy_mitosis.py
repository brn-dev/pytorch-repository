import concurrent.futures
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Iterable

import numpy as np
from gymnasium import Env
from gymnasium.vector import VectorEnv
from overrides import override

from src.datetime import get_current_timestamp
from src.model_db.model_db import ModelDB, ModelEntry
from src.model_db.multiprocessing_sync_wrapper import MultiprocessingSyncWrapper
from src.reinforcement_learning.algorithms.policy_mitosis.policy_mitosis import PolicyMitosis, PolicyWithEnvAndInfo
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policies.policy_initialization import init_policy_using_source
from src.reinforcement_learning.core.policy_info import PolicyInfo
from src.reinforcement_learning.gym.envs.env_wrapping import wrap_env_using_source


class AsyncPolicyMitosis(PolicyMitosis):

    def __init__(
            self,
            num_workers: int,
            policy_db: ModelDB[PolicyInfo],
            policy_train_function: Callable[[PolicyWithEnvAndInfo], tuple[int, float]],
            env_fn: Callable[[], VectorEnv],
            new_init_policy_function: Callable[[], BasePolicy],
            new_wrap_env_function: Callable[[Env | VectorEnv], Env | VectorEnv],
            select_policy_selection_probs: Callable[[Iterable[PolicyInfo]], np.ndarray],
            min_base_ancestors: int,
            rng_seed: int | None,
            _globals: dict[str, Any],
    ):
        super().__init__(
            policy_db=MultiprocessingSyncWrapper(policy_db),
            policy_train_function=policy_train_function,
            env=[env_fn() for _ in range(num_workers)],
            new_init_policy_function=new_init_policy_function,
            new_wrap_env_function=new_wrap_env_function,
            select_policy_selection_probs=select_policy_selection_probs,
            min_base_ancestors=min_base_ancestors,
            rng_seed=rng_seed,
            _globals=_globals,
        )

        self.num_workers = num_workers

        self.num_iterations_started = 0
        self.num_iterations_started_lock = multiprocessing.Lock()

    @override
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

    @override
    def train_lineage(self, start_policy_policy_id: str, nr_iterations: int):
        print(f'Warning: calling train_lineage in AsyncPolicyMitosis - train_lineage can not be parallelized')
        super().train_lineage(
            start_policy_policy_id=start_policy_policy_id,
            nr_iterations=nr_iterations,
        )

