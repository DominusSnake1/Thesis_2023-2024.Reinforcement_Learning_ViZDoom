from Models.Doom_Models import Doom_Models
from Other import Utils


class Doom_Levels:
    def __init__(self):
        self.level = Utils.level_selector()
        self.mode = Utils.mode_selector()
        self.algorithm = Utils.algorithm_selector()

        if self.mode == 'test':
            self.model, self.episodes = Utils.modelAndEpisodes_selector()

    def basic(self):
        """
        Loads the `basic` level in either Training or "Testing mode"
        as well as give a simple description of the level.
        """
        print("\nDescription:\n"
              "The map is a rectangle with gray walls, ceiling, and floor. The player is spawned along the longer "
              "wall in the center. A red, circular monster is spawned randomly somewhere along the opposite wall. A "
              "player can only (config) go left/right and shoot. 1 hit is enough to kill the monster. The episode "
              "finishes when the monster is killed or on timeout.\n")

        model = Doom_Models(level=self.level)

        if self.mode == 'train':
            policy_args = None
            timesteps = 100000

            if self.algorithm == 'PPO':
                policy_args = {'learning_rate': 0.0001, 'n_steps': 2048}

            model.myTrain(algorithm=self.algorithm, total_timesteps=timesteps, policy_used='CnnPolicy',
                          arguments=policy_args)
            return

        if self.mode == 'test':
            model.myTest(algorithm=self.algorithm, model_num=self.model, episodes=self.episodes)
            return

    def defend_the_center(self):
        """
        Loads the `defend_the_center` level in either "Training" or "Testing" mode
        as well as give a simple description of the level.
        """
        print("\nDescription:\n"
              "The map is a large circle. A player is spawned in the exact center. 5 melee-only, monsters are spawned "
              "along the wall. Monsters are killed after a single shot. After dying, each monster is respawned after "
              "some time. The episode ends when the player dies (it’s inevitable because of limited ammo).\n")

        model = Doom_Models(level=self.level)

        if self.mode == 'train':
            policy_args = None
            timesteps = 100000

            if self.algorithm == 'PPO':
                policy_args = {'learning_rate': 0.00001, 'n_steps': 2048}

            model.myTrain(algorithm=self.algorithm, total_timesteps=timesteps, policy_used='CnnPolicy',
                          arguments=policy_args)
            return

        if self.mode == 'test':
            model.myTest(algorithm=self.algorithm, model_num=self.model, episodes=self.episodes)
            return

    def deadly_corridor(self):
        print("\nDescription:\n"
              "The map is a corridor with shooting monsters on both sides (6 monsters in total). A green vest is "
              "placed at the opposite end of the corridor. The reward is proportional (negative or positive) to the "
              "change in the distance between the player and the vest. If the player ignores monsters on the sides "
              "and runs straight for the vest, he will be killed somewhere along the way.\n")

        model = Doom_Models(level=self.level, adjustments=True)

        if self.mode == 'train':
            policy_args = None
            policy = None
            timesteps = 200000

            if self.algorithm == 'PPO':
                policy = 'CnnPolicy'
                policy_args = {'learning_rate': 0.00001, 'n_steps': 4096,
                               'clip_range': 0.1, 'gamma': 0.95, 'gae_lambda': 0.9}
            elif self.algorithm == 'DQN':
                policy = 'CnnPolicy'
                policy_args = {'learning_rate': 0.0001, 'buffer_size': 10_000, 'batch_size': 32}

            model.myTrain(algorithm=self.algorithm, total_timesteps=timesteps, policy_used=policy,
                          arguments=policy_args)
            return

        if self.mode == 'test':
            model.myTest(algorithm=self.algorithm, model_num=self.model, episodes=self.episodes)
            return
