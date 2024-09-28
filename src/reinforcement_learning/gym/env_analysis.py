from typing import Union

from gymnasium import Env, spaces
from gymnasium.vector import VectorEnv


def is_vector_env(env: Env) -> bool:
    try:
        return env.get_wrapper_attr("is_vector_env")
    except AttributeError:
        return False


def get_num_envs(env: Env) -> int:
    try:
        return env.get_wrapper_attr("num_envs")
    except AttributeError:
        return 1


def get_single_obs_space(env: Env) -> spaces.Space:
    if is_vector_env(env):
        return env.get_wrapper_attr('single_observation_space')
    else:
        return env.observation_space


def get_single_action_space(env: Env) -> spaces.Space:
    if is_vector_env(env):
        return env.get_wrapper_attr('single_action_space')
    else:
        return env.action_space


# Adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/preprocessing.py
def get_obs_shape(
        env: Env
) -> Union[tuple[int, ...], dict[str, tuple[int, ...]]]:
    observation_space = get_single_obs_space(env)

    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}

    else:
        raise NotImplementedError(f'{observation_space} observation space is not supported')


# Adapted from https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/preprocessing.py
def get_action_shape(env: Env) -> tuple[int, ...]:
    action_space = get_single_action_space(env)

    if isinstance(action_space, spaces.Box):
        return action_space.shape
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return (1,)
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return (len(action_space.nvec),)
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return (action_space.n,)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")

def get_unique_env_ids(env: VectorEnv) -> list[str]:
    return list(set([s.id for s in env.get_attr('spec')]))

IMPORTANT_SPEC_ATTRIBUTES = ['id', 'kwargs', 'max_episode_steps', 'additional_wrappers']
def get_env_specs(env: VectorEnv):
    return [
        {
            attr: getattr(s, attr)
            for attr in IMPORTANT_SPEC_ATTRIBUTES
        }
        for s in env.get_attr('spec')
    ]
