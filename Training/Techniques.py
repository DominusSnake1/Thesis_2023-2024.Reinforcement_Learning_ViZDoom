import Training.ResNet50


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


class PPO_ResNet:
    """
    This PPO model uses a custom ResNet for feature extraction.
    """

    def __init__(self,
                 policy=Training.ResNet50.CustomCnnPolicy,
                 learning_rate: float = 0.0001,
                 n_steps: int = 4096,
                 ent_coef: float = 0.0001,
                 features_dim: int = 512):

        self.policy = policy
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.algorithm = 'PPO-RN'
        self.reward_shaping = False
        self.curriculum_learning = False
        self.features_dim = features_dim

    def get_policy_kwargs(self):
        from Training.ResNet50 import ResNet50FeatureExtractor

        return dict(
            features_extractor_class=ResNet50FeatureExtractor,
            features_extractor_kwargs={'features_dim': self.features_dim},
        )
