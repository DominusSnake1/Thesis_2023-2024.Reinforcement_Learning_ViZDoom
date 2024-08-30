import Training.CNNFeatureExtractor


class BASE_PPO:
    def __init__(self,
                 number_of_actions: int,
                 doom_skill: int,
                 learning_rate: float = 0.0003,
                 n_steps: int = 4096,
                 ent_coef: float = 0.0,
                 batch_size: int = 64,
                 clip_range: float = 0.2,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        """
        :param number_of_actions: The number of possible actions the agent can take.
        :param learning_rate: Learning rate for the optimizer.
        :param n_steps: The number of steps to run for each environment per update.
        :param ent_coef: Entropy coefficient for the loss calculation.
        :param batch_size: The size of the mini-batch.
        :param clip_range:
        :param gamma:
        :param gae_lambda:
        """
        self.doom_skill = doom_skill

        # PPO Parameters.
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef
        self.clip_range = clip_range
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Policy Parameters
        self.policy = "CnnPolicy"
        self.policy_kwargs = None
        self.number_of_actions = number_of_actions
        self.batch_size = batch_size

        # Reward shaping is not used.
        self.reward_shaping = False

        # Curriculum learning is not used.
        self.curriculum_learning = False


class PPO_Standard(BASE_PPO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Algorithm name for identification.
        self.algorithm = "PPO-S"


class PPO_RewardShaping(BASE_PPO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Algorithm name for identification.
        self.algorithm = "PPO-RS"

        # Reward shaping is used.
        self.reward_shaping = True


class PPO_Curriculum(BASE_PPO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Algorithm name for identification.
        self.algorithm = "PPO-CL"

        # Curriculum learning is not used.
        self.curriculum_learning = True


class PPO_RewardShaping_and_Curriculum(BASE_PPO):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.algorithm = "PPO-RW-CL"

        self.reward_shaping = True
        self.curriculum_learning = True


class BASE_DQN:
    def __init__(self,
                 number_of_actions: int,
                 learning_rate: float = 0.0001,
                 buffer_size: int = 1_000_000,
                 learning_starts: int = 50_000,
                 batch_size: int = 32,
                 tau: float = 1.0,
                 gamma: float = 0.99,
                 gradient_steps: int = 1,
                 exploration_fraction: float = 0.1,
                 exploration_initial_eps: float = 1.0,
                 exploration_final_eps: float = 0.05,
                 max_grad_norm: float = 10):
        self.number_of_actions = number_of_actions
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.exploration_fraction = exploration_fraction
        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.max_grad_norm = max_grad_norm

        # Policy used by the model.
        self.policy = "CnnPolicy"

        # Reward shaping is not used.
        self.reward_shaping = False

        # Curriculum learning is not used.
        self.curriculum_learning = False


class DQN_Standard(BASE_DQN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Algorithm name for identification.
        self.algorithm = "DQN-S"
