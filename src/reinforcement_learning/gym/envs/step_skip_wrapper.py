from typing import SupportsFloat, Any

from gymnasium import Wrapper, Env
from gymnasium.core import WrapperActType, WrapperObsType


class StepSkipWrapper(Wrapper):

    def __init__(self, env: Env, steps_per_step: int):
        super().__init__(env)

        if steps_per_step < 1:
            raise ValueError(f'{steps_per_step = } has to be positive')

        self.skip_steps = steps_per_step

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, total_reward, terminated, truncated, step_info = self.env.step(action)

        info = {'#0': step_info}

        if terminated or truncated:
            return obs, total_reward, terminated, truncated, info

        for skip_step in range(1, self.skip_steps):
            obs, step_reward, terminated, truncated, step_info = self.env.step(action)

            total_reward += step_reward
            info[f'#{skip_step}'] = step_info

            if terminated or truncated:
                return obs, total_reward, terminated, truncated, info

        return obs, total_reward, terminated, truncated, info

