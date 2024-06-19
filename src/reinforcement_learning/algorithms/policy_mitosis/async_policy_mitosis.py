import multiprocessing as mp
import multiprocessing.connection as mp_conn
import sys
import time
import traceback
from typing import Callable, Iterable, Any, Optional

import numpy as np
from gymnasium import Env

from src.model_db.model_db import ModelDB, StateDict
from src.multiprocessing_utils import CloudpickleFunctionWrapper
from src.reinforcement_learning.algorithms.policy_mitosis.mitosis_policy_info import MitosisPolicyInfo
from src.reinforcement_learning.algorithms.policy_mitosis.policy_mitosis_base import PolicyMitosisBase, \
    TrainPolicyFunction, TrainInfo
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policy_construction import PolicyInitializationInfo, PolicyConstruction
from src.torch_device import optimizer_to_device


ModifyPolicyFunction = Callable[[BasePolicy, MitosisPolicyInfo], tuple[BasePolicy, MitosisPolicyInfo]]


class AsyncPolicyMitosis(PolicyMitosisBase):

    def __init__(
            self,
            num_workers: int,
            policy_db: ModelDB[MitosisPolicyInfo],
            train_policy_function: TrainPolicyFunction,
            create_env: Callable[[], Env],
            new_policy_initialization_info: PolicyInitializationInfo,
            new_policy_prob_function: Callable[[int, int], float],
            modify_policy: ModifyPolicyFunction | None,
            select_policy_selection_probs: Callable[[Iterable[MitosisPolicyInfo]], np.ndarray],
            min_primordial_ancestors: int,
            save_optimizer_state_dicts: bool,
            load_optimizer_state_dicts: bool,
            rng_seed: int | None,
            initialization_delay: float = 0.0,
            delay_between_workers: float = 0.0,
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
        self.num_workers = num_workers
        self.create_env = create_env
        self.modify_policy = modify_policy
        self.initialization_delay = initialization_delay
        self.delay_between_workers = delay_between_workers

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
                    CloudpickleFunctionWrapper(self.modify_policy) if self.modify_policy is not None else None,
                    CloudpickleFunctionWrapper(self.train_policy_function),
                    error_queue,
                    self.save_optimizer_state_dicts,
                )
            )

            parent_pipes.append(parent_pipe)
            processes.append(process)

            process.start()
            child_pipe.close()

        if self.initialization_delay > 0:
            time.sleep(self.initialization_delay)

        iterations_started = 0
        for i, pipe in enumerate(parent_pipes):
            iterations_started += 1

            if self.delay_between_workers <= 0:
                self.start_iteration(pipe)
            else:
                delay = self.delay_between_workers * i
                print(f'Starting worker {i} with {delay = }')
                self.start_iteration(pipe)
                time.sleep(self.delay_between_workers)

        while parent_pipes:
            for ready_pipe in mp_conn.wait(parent_pipes):
                success: bool
                policy_state_dict: StateDict
                optimizer_state_dict: Optional[StateDict]
                policy_info: MitosisPolicyInfo

                success, policy_state_dict, optimizer_state_dict, policy_info = ready_pipe.recv()

                if success:
                    print(f'Finished training iteration for policy: {policy_info["policy_id"]}'
                          f', score = {policy_info["score"]}')

                    self.policy_db.save_state_dict(
                        model_id=policy_info['policy_id'],
                        parent_model_id=policy_info['parent_policy_id'],
                        model_info=policy_info,
                        model_state_dict=policy_state_dict,
                        optimizer_state_dict=optimizer_state_dict if self.save_optimizer_state_dicts else None
                    )
                else:
                    error_items = error_queue.get()
                    print(error_items[:-1])
                    print(error_items[-1])

                if iterations_started < nr_iterations:
                    self.start_iteration(ready_pipe)
                    iterations_started += 1
                else:
                    ready_pipe.send((False, None, None, None))
                    ready_pipe.close()
                    parent_pipes.remove(ready_pipe)

        for process in processes:
            process.join()

        print(f'Async Policy Mitosis finished - completed {iterations_started} iterations')

    def start_iteration(self, pipe: mp_conn.Connection):
        policy_info = self.pick_policy_info()

        if policy_info['parent_policy_id'] is not None:
            policy_state_dict, optimizer_state_dict = self.policy_db.load_state_dict(
                policy_info['parent_policy_id'],
                load_optimizer=self.load_optimizer_state_dicts
            )
        else:
            policy_state_dict, optimizer_state_dict = None, None

        pipe.send((True, policy_state_dict, optimizer_state_dict, policy_info))
        print(f'Started training iteration for policy: {policy_info["policy_id"]}, '
              f'parent policy id: {policy_info["parent_policy_id"]}')


def _worker(
        index: int,
        pipe: mp_conn.Connection,
        parent_pipe: mp_conn.Connection,
        create_env: Callable[[], Env],
        modify_policy: ModifyPolicyFunction | None,
        train_policy_function: TrainPolicyFunction,
        error_queue: mp.Queue,
        send_optimizer_state_dict: bool
):
    parent_pipe.close()
    env = create_env()

    try:
        while True:
            continue_: bool
            policy_state_dict: dict[str, Any] | None
            optimizer_state_dict: dict[str, Any] | None
            policy_info: MitosisPolicyInfo

            continue_, policy_state_dict, optimizer_state_dict, policy_info = pipe.recv()

            if not continue_:
                break

            policy, optimizer, env = PolicyConstruction.init_and_apply_state_dicts(
                info=policy_info['initialization_info'],
                env=env,
                policy_state_dict=policy_state_dict,
                optimizer_state_dict=optimizer_state_dict,
            )

            if modify_policy is not None:
                policy, policy_info = modify_policy(policy, policy_info)

            train_info: TrainInfo = {
                'policy': policy,
                'optimizer': optimizer,
                'env': env,
                'policy_info': policy_info,
            }

            PolicyMitosisBase.train_policy_iteration(train_policy_function, train_info)

            optimizer_state_dict: Optional[dict[str, Any]] = None
            if send_optimizer_state_dict:
                optimizer_state_dict = optimizer_to_device(optimizer, 'cpu').state_dict()

            pipe.send((True, policy.cpu().state_dict(), optimizer_state_dict, policy_info))
    except (KeyboardInterrupt, Exception) as exc:
        error_queue.put((index,) + sys.exc_info()[:-1] + (traceback.format_exc(),))
        sys.stderr.write(str(exc))
        sys.stdout.flush()
        pipe.send((False, None, None, None))
    finally:
        env.close()

