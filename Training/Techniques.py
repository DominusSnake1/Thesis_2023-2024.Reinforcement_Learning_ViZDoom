import Training.ResNet50


class PPO_Standard:
    """
    The standard PPO technique for comparison. The base model with no changes.
    """

    def __init__(self,
                 learning_rate=0.0001,
                 n_steps=4096,
                 ent_coef=0.001):
        self.algorithm = 'PPO-S'
        self.policy = 'CnnPolicy'
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
        self.policy = 'CnnPolicy'
        self.reward_shaping = True
        self.curriculum_learning = False
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef


class PPO_ResNet:
    """
    This PPO model uses a custom ResNet for feature extraction.
    """

    def __init__(self,
                 learning_rate=0.0001,
                 n_steps=2048,
                 ent_coef=0.001,
                 clip_range=0.2,
                 features_dim=512):
        self.algorithm = 'PPO-RN'
        self.policy = Training.ResNet50.CustomCnnPolicy
        self.reward_shaping = False
        self.curriculum_learning = False
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.features_dim = features_dim

    def get_policy_kwargs(self):
        from Training.ResNet50 import ResNet50FeatureExtractor

        return dict(
            features_extractor_class=ResNet50FeatureExtractor,
            features_extractor_kwargs={'features_dim': self.features_dim},
        )
