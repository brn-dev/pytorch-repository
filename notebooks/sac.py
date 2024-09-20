from typing import Any

import gymnasium
import torch

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policy_construction import InitActionSelectorFunction


def init_action_selector(latent_dim: int, action_dim: int, hyper_parameters: dict[str, 'Any']) -> 'ActionSelector':
    from src.reinforcement_learning.core.action_selectors.predicted_std_action_selector \
        import PredictedStdActionSelector
    from src.reinforcement_learning.core.action_selectors.squashed_diag_gaussian_action_selector import \
        SquashedDiagGaussianActionSelector
    from src.weight_initialization import orthogonal_initialization

    return PredictedStdActionSelector(
        latent_dim=latent_dim,
        action_dim=action_dim,
        base_std=1.0,
        squash_output=True,
        action_net_initialization=lambda module: orthogonal_initialization(module, gain=0.01),
        log_std_net_initialization=lambda module: orthogonal_initialization(module, gain=0.1),
    )
    # return PredictedStdActionSelector(
    #     latent_dim=latent_dim,
    #     action_dim=action_dim,
    #     std=1.0,
    #     std_learnable=True,
    #     action_net_initialization=lambda module: orthogonal_initialization(module, gain=0.01),
    # )



def init_policy(
        init_action_selector_: 'InitActionSelectorFunction',
        hyper_parameters: dict[str, 'Any']
) -> 'BasePolicy':
    import torch
    from torch import nn
    import numpy as np

    from src.reinforcement_learning.algorithms.sac.sac_policy import SACPolicy
    from src.networks.core.seq_net import SeqNet
    from src.networks.core.net import Net
    from src.networks.skip_nets.additive_skip_connection import AdditiveSkipConnection
    from src.weight_initialization import orthogonal_initialization
    from src.reinforcement_learning.core.policies.components.actor import Actor
    from src.reinforcement_learning.core.policies.components.q_critic import QCritic

    # in_size = 376
    # action_size = 17
    # actor_out_sizes = [512, 512, 256, 256, 256, 256, 256, 256]
    # critic_out_sizes = [512, 512, 256, 256, 256, 1]

    in_size = 17
    action_size = 6

    actor_layers = 3
    actor_features = 96

    critic_layers = 2
    critic_features = 96

    hidden_activation_function = nn.ELU

    # actor_net = nn.Sequential(
    #     nn.Linear(in_size, actor_features),
    #     hidden_activation_function(),
    #     SeqNet.from_layer_provider(
    #         layer_provider=lambda layer_nr, is_last_layer, in_features, out_features:
    #         AdditiveSkipConnection(Net.seq_as_net(
    #             orthogonal_initialization(nn.Linear(in_features, out_features), gain=np.sqrt(2)),
    #             hidden_activation_function(),
    #             orthogonal_initialization(nn.Linear(in_features, out_features), gain=np.sqrt(2)),
    #             nn.Tanh() if is_last_layer else hidden_activation_function(),
    #         )),
    #         num_features=actor_features,
    #         num_layers=actor_layers,
    #     )
    # )
    #
    # critic = QCritic(
    #     n_critics=2,
    #     create_q_network=lambda: nn.Sequential(
    #         nn.Linear(in_size + action_size, critic_features),
    #         hidden_activation_function(),
    #         SeqNet.from_layer_provider(
    #             layer_provider=lambda layer_nr, is_last_layer, in_features, out_features:
    #             AdditiveSkipConnection(Net.seq_as_net(
    #                 orthogonal_initialization(nn.Linear(in_features, out_features), gain=np.sqrt(2)),
    #                 hidden_activation_function(),
    #                 orthogonal_initialization(nn.Linear(in_features, out_features), gain=np.sqrt(2)),
    #                 hidden_activation_function(),
    #             )),
    #             num_features=critic_features,
    #             num_layers=critic_layers,
    #         ),
    #         nn.Linear(critic_features, 1)
    #     )
    # )

    actor_net = nn.Sequential(
        nn.Linear(in_size, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
    )

    critic = QCritic(
        n_critics=2,
        create_q_network=lambda: nn.Sequential(
            nn.Linear(in_size + action_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    )

    return SACPolicy(
        actor=Actor(actor_net, init_action_selector_(
            latent_dim=256,
            action_dim=action_size,
            hyper_parameters=hyper_parameters,
        )),
        critic=critic
    )

def init_optimizer(pol: 'BasePolicy', hyper_parameters: dict[str, 'Any']) -> 'torch.optim.Optimizer':
    import torch.optim
    return torch.optim.AdamW(pol.parameters(), lr=3e-4, weight_decay=1e-4)

def wrap_env(env_: 'gymnasium.vector.VectorEnv', hyper_parameters: dict[str, 'Any']) -> 'gymnasium.Env':
    from src.reinforcement_learning.gym.wrappers.transform_reward_wrapper import TransformRewardWrapper
    from gymnasium.wrappers import RescaleAction

    env_ = TransformRewardWrapper(env_, lambda reward_: 0.5 * reward_)
    env_ = RescaleAction(env_, min_action=-1.0, max_action=1.0)
    return env_
