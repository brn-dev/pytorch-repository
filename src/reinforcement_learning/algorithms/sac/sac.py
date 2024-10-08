from dataclasses import dataclass
from typing import Type, Optional, Any, Literal

import gymnasium
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from src.function_types import TorchTensorFn
from src.module_analysis import calculate_grad_norm
from src.hyper_parameters import HyperParameters
from src.reinforcement_learning.algorithms.base.base_algorithm import PolicyProvider
from src.reinforcement_learning.algorithms.base.off_policy_algorithm import OffPolicyAlgorithm, ReplayBuf
from src.reinforcement_learning.algorithms.sac.sac_crossq_policy import SACCrossQPolicy
from src.reinforcement_learning.algorithms.sac.sac_policy import SACPolicy
from src.reinforcement_learning.core.action_noise import ActionNoise
from src.reinforcement_learning.core.buffers.replay.base_replay_buffer import BaseReplayBuffer, ReplayBufferSamples
from src.reinforcement_learning.core.buffers.replay.replay_buffer import ReplayBuffer
from src.reinforcement_learning.core.callback import Callback
from src.reinforcement_learning.core.infos import InfoDict, concat_infos
from src.reinforcement_learning.core.logging import LoggingConfig, log_if_enabled
from src.reinforcement_learning.core.loss_config import weigh_and_reduce_loss, LossLoggingConfig
from src.reinforcement_learning.core.type_aliases import OptimizerProvider, TensorObs, detach_obs
from src.reinforcement_learning.gym.env_analysis import get_single_action_space
from src.tags import Tags
from src.torch_device import TorchDevice
from src.torch_functions import identity
from src.repr_utils import func_repr

SAC_DEFAULT_OPTIMIZER_PROVIDER = lambda params: optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
AUTO_TARGET_ENTROPY = 'auto'


@dataclass
class SACLoggingConfig(LoggingConfig):

    log_entropy_coef: bool = False
    entropy_coef_loss: LossLoggingConfig = None
    actor_loss: LossLoggingConfig = None
    critic_loss: LossLoggingConfig = None

    def __post_init__(self):
        if self.actor_loss is None:
            self.actor_loss = LossLoggingConfig()
        if self.entropy_coef_loss is None:
            self.entropy_loss = LossLoggingConfig()
        if self.critic_loss is None:
            self.critic_loss = LossLoggingConfig()

        super().__post_init__()


"""

        Soft Actor-Critic:
        Off-Policy Maximum Entropy Deep Reinforcement
        Learning with a Stochastic Actor
        https://arxiv.org/pdf/1801.01290

"""
class SAC(OffPolicyAlgorithm[SACPolicy, ReplayBuf, SACLoggingConfig]):

    buffer: BaseReplayBuffer
    target_entropy: float
    log_ent_coef: Optional[torch.Tensor]
    entropy_coef_optimizer: Optional[optim.Optimizer]
    entropy_coef_tensor: Optional[torch.Tensor]

    def __init__(
            self,
            env: gymnasium.Env,
            policy: SACPolicy | PolicyProvider[SACPolicy],
            actor_optimizer_provider: OptimizerProvider = SAC_DEFAULT_OPTIMIZER_PROVIDER,
            critic_optimizer_provider: OptimizerProvider = SAC_DEFAULT_OPTIMIZER_PROVIDER,
            weigh_and_reduce_actor_loss: TorchTensorFn = torch.mean,
            weigh_critic_loss: TorchTensorFn = identity,
            buffer_type: Type[ReplayBuf] = ReplayBuffer,
            buffer_size: int = 100_000,
            buffer_kwargs: dict[str, Any] = None,
            reward_scale: float = 1.0,
            gamma: float = 0.99,
            tau: float = 0.005,
            rollout_steps: int = 1,
            gradient_steps: int = 1,
            optimization_batch_size: int = 256,
            target_update_interval: int = 1,
            entropy_coef: float = 1.0,
            target_entropy: float | Literal['auto'] = AUTO_TARGET_ENTROPY,
            entropy_coef_optimizer_provider: Optional[OptimizerProvider] = None,
            weigh_and_reduce_entropy_coef_loss: TorchTensorFn = torch.mean,
            action_noise: Optional[ActionNoise] = None,
            warmup_steps: int = 100,
            sde_noise_sample_freq: Optional[int] = None,
            callback: Callback['SAC'] = None,
            logging_config: SACLoggingConfig = None,
            torch_device: TorchDevice = 'auto',
            torch_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(
            env=env,
            policy=policy,
            buffer=buffer_type.for_env(
                env=env,
                buffer_size=buffer_size,
                torch_device=torch_device,
                torch_dtype=torch_dtype,
                reward_scale=reward_scale,
                **(buffer_kwargs or {})
            ),
            gamma=gamma,
            tau=tau,
            rollout_steps=rollout_steps,
            gradient_steps=gradient_steps,
            optimization_batch_size=optimization_batch_size,
            action_noise=action_noise,
            warmup_steps=warmup_steps,
            sde_noise_sample_freq=sde_noise_sample_freq,
            callback=callback or Callback(),
            logging_config=logging_config or LoggingConfig(),
            torch_device=torch_device,
            torch_dtype=torch_dtype,
        )

        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.shared_feature_extractor = self.policy.shared_feature_extractor

        self.actor_optimizer = actor_optimizer_provider(
            self.chain_parameters(self.actor, self.shared_feature_extractor)
        )
        self.critic_optimizer = critic_optimizer_provider(self.critic.parameters())

        self.weigh_and_reduce_entropy_coef_loss = weigh_and_reduce_entropy_coef_loss
        self.weigh_and_reduce_actor_loss = weigh_and_reduce_actor_loss
        self.weigh_critic_loss = weigh_critic_loss

        self.target_update_interval = target_update_interval
        self.gradient_steps_performed = 0

        self._setup_entropy_optimization(entropy_coef, target_entropy, entropy_coef_optimizer_provider)

        # CrossQ doesn't use a target critic
        if isinstance(self.policy, SACCrossQPolicy):
            self.tau = 0
            self.target_update_interval = 0

    def collect_hyper_parameters(self) -> HyperParameters:
        return self.update_hps(super().collect_hyper_parameters(), {
            'actor_optimizer': self.get_optimizer_hps(self.actor_optimizer),
            'critic_optimizer': self.get_optimizer_hps(self.critic_optimizer),
            'entropy_coef_optimizer': self.maybe_get_optimizer_hps(self.entropy_coef_optimizer),
            'weigh_and_reduce_entropy_coef_loss': self.maybe_get_func_repr(self.weigh_and_reduce_entropy_coef_loss),
            'weigh_and_reduce_actor_loss': self.get_func_repr(self.weigh_and_reduce_actor_loss),
            'weigh_critic_loss': self.get_func_repr(self.weigh_critic_loss),
            'target_update_interval': self.target_update_interval,
            'target_entropy': self.target_entropy,
            'entropy_coef': self.entropy_coef_tensor.item() if self.entropy_coef_tensor is not None else 'dynamic',
        })

    def _setup_entropy_optimization(
            self,
            entropy_coef: float,
            target_entropy: float | Literal['auto'],
            entropy_coef_optimizer_provider: Optional[OptimizerProvider],
    ):
        if target_entropy == 'auto':
            self.target_entropy = float(-np.prod(get_single_action_space(self.env).shape).astype(np.float32))
        else:
            self.target_entropy = float(target_entropy)

        if entropy_coef_optimizer_provider is not None:
            self.log_ent_coef = torch.log(
                torch.tensor([entropy_coef], device=self.torch_device, dtype=self.torch_dtype)
            ).requires_grad_(True)
            self.entropy_coef_optimizer = entropy_coef_optimizer_provider([self.log_ent_coef])
            self.entropy_coef_tensor = None
        else:
            self.log_ent_coef = None
            self.entropy_coef_optimizer = None
            self.entropy_coef_tensor = torch.tensor(entropy_coef, device=self.torch_device, dtype=self.torch_dtype)

    def get_and_optimize_entropy_coef(
            self,
            actions_pi_log_prob: torch.Tensor,
            info: InfoDict
    ) -> torch.Tensor:
        if self.entropy_coef_optimizer is not None:
            entropy_coef = torch.exp(self.log_ent_coef.detach())

            entropy_coef_loss = weigh_and_reduce_loss(
                raw_loss=-self.log_ent_coef * (actions_pi_log_prob + self.target_entropy).detach(),
                weigh_and_reduce_function=self.weigh_and_reduce_entropy_coef_loss,
                info=info,
                loss_name='entropy_coef_loss',
                logging_config=self.logging_config.entropy_coef_loss
            )
            self.entropy_coef_optimizer.zero_grad()
            entropy_coef_loss.backward()
            self.entropy_coef_optimizer.step()

            return entropy_coef
        else:
            return self.entropy_coef_tensor

    def calculate_critic_loss(
            self,
            observation_features: TensorObs,
            replay_samples: ReplayBufferSamples,
            entropy_coef: torch.Tensor,
            info: InfoDict,
    ):
        target_q_values = self.policy.compute_target_values(
            replay_samples=replay_samples,
            entropy_coef=entropy_coef,
            gamma=self.gamma,
        )
        # critic loss should not influence shared feature extractor
        current_q_values = self.critic(detach_obs(observation_features), replay_samples.actions)

        # noinspection PyTypeChecker
        critic_loss: torch.Tensor = 0.5 * sum(
            F.mse_loss(current_q, target_q_values) for current_q in current_q_values
        )
        critic_loss = weigh_and_reduce_loss(
            raw_loss=critic_loss,
            weigh_and_reduce_function=self.weigh_critic_loss,
            info=info,
            loss_name='critic_loss',
            logging_config=self.logging_config.critic_loss,
        )
        return critic_loss

    def calculate_actor_loss(
            self,
            observation_features: TensorObs,
            actions_pi: torch.Tensor,
            actions_pi_log_prob: torch.Tensor,
            entropy_coef: torch.Tensor,
            info: InfoDict,
    ) -> torch.Tensor:
        q_values_pi = torch.cat(self.critic(observation_features, actions_pi), dim=-1)
        min_q_values_pi, _ = torch.min(q_values_pi, dim=-1, keepdim=True)
        actor_loss = entropy_coef * actions_pi_log_prob - min_q_values_pi

        actor_loss = weigh_and_reduce_loss(
            raw_loss=actor_loss,
            weigh_and_reduce_function=self.weigh_and_reduce_actor_loss,
            info=info,
            loss_name='actor_loss',
            logging_config=self.logging_config.actor_loss,
        )

        return actor_loss

    def optimize(self, last_obs: np.ndarray, last_episode_starts: np.ndarray, info: InfoDict) -> None:
        gradient_step_infos: list[InfoDict] = []

        for gradient_step in range(self.gradient_steps):
            step_info: InfoDict = {}
            replay_samples = self.buffer.sample(self.optimization_batch_size)
            
            # TODO!!!
            # self.actor.reset_sde_noise()  # TODO: set batch size?

            observation_features = self.shared_feature_extractor(replay_samples.observations)
            # TODO!!!
            actions_pi, actions_pi_log_prob = self.actor.action_log_prob(observation_features)
            actions_pi_log_prob = actions_pi_log_prob.reshape(-1, 1)

            entropy_coef = self.get_and_optimize_entropy_coef(actions_pi_log_prob, step_info)
            log_if_enabled(step_info, 'entropy_coef', entropy_coef, self.logging_config.log_entropy_coef)

            critic_loss = self.calculate_critic_loss(
                observation_features=observation_features,
                replay_samples=replay_samples,
                entropy_coef=entropy_coef,
                info=step_info
            )

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = self.calculate_actor_loss(
                observation_features=observation_features,
                actions_pi=actions_pi,
                actions_pi_log_prob=actions_pi_log_prob,
                entropy_coef=entropy_coef,
                info=step_info
            )

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.gradient_steps_performed += 1
            if self.target_update_interval > 0 and self.gradient_steps_performed % self.target_update_interval == 0:
                self.policy.perform_polyak_update(self.tau)

            gradient_step_infos.append(step_info)
        info.update(concat_infos(gradient_step_infos))



