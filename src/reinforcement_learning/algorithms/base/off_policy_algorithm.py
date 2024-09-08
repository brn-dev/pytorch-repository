from abc import ABC
from typing import TypeVar

import gymnasium
import torch

from src.reinforcement_learning.algorithms.base.base_algorithm import BaseAlgorithm, Policy, LogConf, PolicyProvider
from src.reinforcement_learning.algorithms.base.on_policy_algorithm import RolloutBuf
from src.reinforcement_learning.core.action_noise import ActionNoise
from src.reinforcement_learning.core.buffers.replay.base_replay_buffer import BaseReplayBuffer
from src.reinforcement_learning.core.callback import Callback
from src.torch_device import TorchDevice

ReplayBuf = TypeVar('ReplayBuf', bound=BaseReplayBuffer)


class OffPolicyAlgorithm(BaseAlgorithm[Policy, ReplayBuf, LogConf], ABC):

    def __init__(
            self,
            env: gymnasium.Env,
            policy: Policy | PolicyProvider,
            buffer: RolloutBuf,
            gamma: float,
            tau: float,
            rollout_steps: int,
            gradient_steps: int,
            action_noise: ActionNoise,
            warmup_steps: int,
            sde_noise_sample_freq: int | None,
            use_sde_during_warmup: bool,
            callback: Callback,
            logging_config: LogConf,
            torch_device: TorchDevice,
            torch_dtype: torch.dtype
    ):
        super().__init__(
            env=env,
            policy=policy,
            buffer=buffer,
            gamma=gamma,
            sde_noise_sample_freq=sde_noise_sample_freq,
            callback=callback,
            logging_config=logging_config,
            torch_device=torch_device,
            torch_dtype=torch_dtype,
        )

        self.tau = tau

        self.rollout_steps = rollout_steps
        self.gradient_steps = gradient_steps

        self.action_noise = action_noise
        self.warmup_steps = warmup_steps
        self.use_sde_during_warmup = use_sde_during_warmup

