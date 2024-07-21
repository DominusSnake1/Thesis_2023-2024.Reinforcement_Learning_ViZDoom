import Training.CNNFeatureExtractor as cnn


class PPO_Standard:
    """
    The standard PPO technique for comparison. The base model with no changes.
    """

    def __init__(self,
                 number_of_actions: int,
                 learning_rate=0.0001,
                 n_steps=4096,
                 ent_coef=0.001):
        self.algorithm = 'PPO-S'
        self.policy = cnn.CustomCnnPolicy
        self.number_of_actions = number_of_actions
        self.reward_shaping = False
        self.curriculum_learning = False
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef


class PPO_RewardShaping:
    """
    This PPO model uses Reward Shaping to generate better results.
    """

    def __init__(self,
                 learning_rate=0.0001,
                 n_steps=4096,
                 ent_coef=0.001):
        self.algorithm = 'PPO-RS'
        self.policy = cnn.CustomCnnPolicy
        self.reward_shaping = True
        self.curriculum_learning = False
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef
