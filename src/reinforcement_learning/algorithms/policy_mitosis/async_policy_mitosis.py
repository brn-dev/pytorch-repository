import multiprocessing as mp
import multiprocessing.connection as mp_conn
import sys
from typing import Callable, Iterable, Any

import numpy as np
from gymnasium import Env
from gymnasium.vector import VectorEnv

from src.model_db.model_db import ModelDB
from src.multiprocessing_utils import CloudpickleFunctionWrapper
from src.reinforcement_learning.algorithms.policy_mitosis.policy_mitosis_base import PolicyMitosisBase, \
    TrainPolicyFunction, PolicyWithEnvAndInfo
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policies.policy_initialization import init_policy_using_source
from src.reinforcement_learning.algorithms.policy_mitosis.mitosis_policy_info import MitosisPolicyInfo
from src.reinforcement_learning.gym.envs.env_wrapping import wrap_env_using_source


class AsyncPolicyMitosis(PolicyMitosisBase):

    def __init__(
            self,
            num_workers: int,
            policy_db: ModelDB[MitosisPolicyInfo],
            train_policy_function: TrainPolicyFunction,
            create_env: Callable[[], Env],
            new_init_policy_function: Callable[[], BasePolicy],
            new_wrap_env_function: Callable[[Env | VectorEnv], Env | VectorEnv],
            select_policy_selection_probs: Callable[[Iterable[MitosisPolicyInfo]], np.ndarray],
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
        self.num_workers = num_workers
        self.create_env = create_env

    def train_with_mitosis(self, nr_iterations: int):

        parent_pipes: list[mp.Pipe] = []
        processes: list[mp.Process] = []
        error_queue = mp.Queue()

        for i in range(self.num_workers):
            parent_pipe, child_pipe = mp.Pipe()
            process = mp.Process(
                target=_worker,
                name=f'MitosisWorker-{i}',
                args=(
                    i,
                    child_pipe,
                    parent_pipe,
                    CloudpickleFunctionWrapper(self.create_env),
                    CloudpickleFunctionWrapper(self.train_policy_function),
                    error_queue,
                )
            )

            parent_pipes.append(parent_pipe)
            processes.append(process)

            process.start()
            child_pipe.close()

        iterations_started = 0
        for pipe in parent_pipes:
            self.start_iteration(pipe)
            iterations_started += 1

        while parent_pipes:
            for ready_pipe in mp_conn.wait(parent_pipes):
                success: bool
                policy_state_dict: dict[str, Any]
                policy_info: MitosisPolicyInfo

                success, policy_state_dict, policy_info = ready_pipe.recv()

                if success:
                    print(f'Finished training iteration for policy: {policy_info["policy_id"]}'
                          f', score = {policy_info["score"]}')

                    self.policy_db.save_state_dict(
                        state_dict=policy_state_dict,
                        model_id=policy_info['policy_id'],
                        parent_model_id=policy_info['parent_policy_id'],
                        model_info=policy_info
                    )
                else:
                    print(error_queue.get())

                if iterations_started < nr_iterations:
                    self.start_iteration(ready_pipe)
                    iterations_started += 1
                else:
                    ready_pipe.send((False, None, None))
                    ready_pipe.close()
                    parent_pipes.remove(ready_pipe)

        for process in processes:
            process.join()

        print(f'Async Policy Mitosis finished - completed {iterations_started} iterations')

    def start_iteration(self, pipe: mp_conn.Connection):
        policy_info = self.pick_policy_info()

        policy_state_dict: dict[str, Any] | None
        if policy_info['parent_policy_id'] is not None:
            policy_state_dict, _ = self.policy_db.load_state_dict(policy_info['parent_policy_id'])
        else:
            policy_state_dict = None

        pipe.send((True, policy_state_dict, policy_info))
        print(f'Started training iteration for policy: {policy_info["policy_id"]}, '
              f'parent policy id: {policy_info["parent_policy_id"]}')


def _worker(
        index: int,
        pipe: mp_conn.Connection,
        parent_pipe: mp_conn.Connection,
        create_env: Callable[[], Env],
        train_policy_function: TrainPolicyFunction,
        error_queue: mp.Queue,
):
    parent_pipe.close()
    env = create_env()

    try:
        while True:
            _continue: bool
            policy_state_dict: dict[str, Any] | None
            policy_info: MitosisPolicyInfo

            _continue, policy_state_dict, policy_info = pipe.recv()

            if not _continue:
                break

            policy = init_policy_using_source(policy_info['init_policy_source_code'])
            if policy_state_dict is not None:
                policy.load_state_dict(policy_state_dict)

            policy_with_env_and_info: PolicyWithEnvAndInfo = {
                'policy': policy,
                'env': wrap_env_using_source(env, policy_info['wrap_env_source_code']),
                'policy_info': policy_info,
            }

            PolicyMitosisBase.train_policy_iteration(train_policy_function, policy_with_env_and_info)

            pipe.send((True, policy.cpu().state_dict(), policy_info))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:])
        pipe.send((False, None, None))
    finally:
        env.close()

