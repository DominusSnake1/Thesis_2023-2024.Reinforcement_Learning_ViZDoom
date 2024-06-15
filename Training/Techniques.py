class PPO_Standard:
    """
    The standard PPO technique for comparison. The base model with no changes.
    """
    def __init__(self,
                 policy: str = 'CnnPolicy',
                 learning_rate: float = 0.0001,
                 n_steps: int = 4096,
                 ent_coef: float = 0.001):

        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.algorithm = 'PPO-S'
        self.reward_shaping = False
        self.curriculum_learning = False


class PPO_RewardShaping:
    """
    This PPO model uses Reward Shaping to generate better results.
    """
    def __init__(self,
                 policy: str = 'CnnPolicy',
                 learning_rate: float = 0.0001,
                 n_steps: int = 4096,
                 ent_coef: float = 0.001):

        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.algorithm = 'PPO-RS'
        self.reward_shaping = True
        self.curriculum_learning = False
