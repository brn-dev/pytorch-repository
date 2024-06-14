import gymnasium


class RewardWrapper(gymnasium.core.Wrapper):
    RAW_REWARDS_KEY = 'raw_rewards'
