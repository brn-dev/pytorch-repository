from src.reinforcement_learning.core.policies.base_policy import BasePolicy
from src.reinforcement_learning.core.type_aliases import TensorObs


class ActorPolicy(BasePolicy):

    def forward(self, obs: TensorObs):
        return self.act(obs)
