class PPO_Standard:
    def __init__(self,
                 number_of_actions: int,
                 learning_rate: float = 0.0001,
                 n_steps: int = 4096,
                 ent_coef: float = 0.001):
        """
        Initializes the PPO standard model.

        :param number_of_actions: The number of possible actions the agent can take.
        :param learning_rate: Learning rate for the optimizer.
        :param n_steps: The number of steps to run for each environment per update.
        :param ent_coef: Entropy coefficient for the loss calculation.
        """
        self.number_of_actions = number_of_actions
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef

        # Algorithm name for identification.
        self.algorithm = "PPO-S"

        # Policy used by the model.
        self.policy = "CnnPolicy"

        # Reward shaping is not used.
        self.reward_shaping = False

        # Curriculum learning is not used.
        self.curriculum_learning = False


class PPO_RewardShaping:
    def __init__(self,
                 number_of_actions: int,
                 learning_rate: float = 0.0001,
                 n_steps: int = 4096,
                 ent_coef: float = 0.001):
        """
        Initializes the PPO RewardShaping model.

        :param number_of_actions: The number of possible actions the agent can take.
        :param learning_rate: Learning rate for the optimizer.
        :param n_steps: The number of steps to run for each environment per update.
        :param ent_coef: Entropy coefficient for the loss calculation.
        """
        self.number_of_actions = number_of_actions
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.ent_coef = ent_coef

        # Algorithm name for identification.
        self.algorithm = "PPO-RS"

        # Policy used by the model.
        self.policy = "CnnPolicy"

        # Reward shaping is used.
        self.reward_shaping = True

        # Curriculum learning is not used.
        self.curriculum_learning = False
