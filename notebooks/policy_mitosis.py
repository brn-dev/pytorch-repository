from typing import Any

import torch
import gymnasium

from src.reinforcement_learning.core.action_selectors.action_selector import ActionSelector
from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.policy_construction import InitActionSelectorFunction

# Type hinting as string, so they don't cause an error when evaluating them outside this context
def init_action_selector(latent_dim: int, action_dim: int, hyper_parameters: dict[str, 'Any']) -> 'ActionSelector':
    from src.reinforcement_learning.core.action_selectors.predicted_std_action_selector \
        import PredictedStdActionSelector
    from src.weight_initialization import orthogonal_initialization

    return PredictedStdActionSelector(
        latent_dim=latent_dim,
        action_dim=action_dim,
        base_std=0.15,
        squash_output=True,
        action_net_initialization=lambda module: orthogonal_initialization(module, gain=0.01),
        log_std_net_initialization=lambda module: orthogonal_initialization(module, gain=0.1),
    )

def init_policy(
        init_action_selector_: 'InitActionSelectorFunction',
        hyper_parameters: dict[str, 'Any']
) -> 'BasePolicy':
    import torch
    from torch import nn

    from src.networks.core.seq_net import SeqNet
    from src.reinforcement_learning.core.policies.actor_critic_policy import ActorCriticPolicy
    from src.networks.core.net import Net
    from src.networks.skip_nets.additive_skip_connection import AdditiveSkipConnection

    in_size = 27
    action_size = 8

    actor_layers = 7
    actor_features = 64

    critic_layers = 4
    critic_features = 64

    hidden_activation_function = nn.ELU

    class A2CNetwork(nn.Module):

        def __init__(self):
            super().__init__()

            self.actor_embedding = nn.Sequential(
                nn.Linear(in_size, actor_features),
                hidden_activation_function()
            )
            self.actor = SeqNet.from_layer_provider(
                layer_provider=lambda layer_nr, is_last_layer, in_features, out_features:
                    AdditiveSkipConnection(Net.seq_as_net(
                        nn.Linear(in_features, out_features),
                        nn.Tanh() if is_last_layer else hidden_activation_function()
                    )),
                num_features=actor_features,
                num_layers=actor_layers,
            )

            self.critic_embedding = nn.Sequential(
                nn.Linear(in_size, critic_features),
                hidden_activation_function()
            )
            self.critic = SeqNet.from_layer_provider(
                layer_provider=lambda layer_nr, is_last_layer, in_features, out_features:
                    AdditiveSkipConnection(Net.seq_as_net(
                        nn.Linear(in_features, out_features),
                        hidden_activation_function()
                    )),
                num_features=critic_features,
                num_layers=critic_layers,
            )
            self.critic_regressor = nn.Linear(critic_features, 1)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return (
                self.actor(self.actor_embedding(x)),
                self.critic_regressor(self.critic(self.critic_embedding(x)))
            )

    return ActorCriticPolicy(
        (network := A2CNetwork()),
        init_action_selector_(
            latent_dim=network.actor.out_shape.get_definite_features(),
            action_dim=action_size,
            hyper_parameters=hyper_parameters,
        )
    )

def init_optimizer(policy: 'BasePolicy', hyper_parameters: dict[str, 'Any']) -> 'torch.optim.Optimizer':
    import torch.optim
    return torch.optim.AdamW(policy.parameters(), lr=1e-5)

def wrap_env(env_: 'gymnasium.vector.VectorEnv', hyper_parameters: dict[str, 'Any']) -> 'gymnasium.Env':
    from src.reinforcement_learning.gym.wrappers.transform_reward_wrapper import TransformRewardWrapper
    from gymnasium.wrappers import RescaleAction

    env_ = TransformRewardWrapper(env_, lambda _reward: 0.01 * _reward)
    env_ = RescaleAction(env_, min_action=-1.0, max_action=1.0)

    return env_
