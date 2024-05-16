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
        print("\nDescription:\n"
              "The map is a rectangle with gray walls, ceiling, and floor. The player is spawned along the longer "
              "wall in the center. A red, circular monster is spawned randomly somewhere along the opposite wall. A "
              "player can only (config) go left/right and shoot. 1 hit is enough to kill the monster. The episode "
              "finishes when the monster is killed or on timeout.\n")

        model = Doom_Models(level=self.level, algorithm=self.algorithm)

        if self.mode == 'train':
            policy_args = None
            timesteps = 100000

            if self.algorithm == 'PPO':
                policy_args = {'learning_rate': 0.0001, 'n_steps': 2048}

            model.myTrain(total_timesteps=timesteps, policy_used='CnnPolicy', arguments=policy_args)
            return

        if self.mode == 'test':
            model.myTest(model_num=self.model, episodes=self.episodes)
            return

    def defend_the_center(self):
        print("\nDescription:\n"
              "The map is a large circle. A player is spawned in the exact center. 5 melee-only, monsters are spawned "
              "along the wall. Monsters are killed after a single shot. After dying, each monster is respawned after "
              "some time. The episode ends when the player dies (it’s inevitable because of limited ammo).\n")

        model = Doom_Models(level=self.level, algorithm=self.algorithm)

        if self.mode == 'train':
            policy_args = None
            timesteps = 100000

            if self.algorithm == 'PPO':
                policy_args = {'learning_rate': 0.00001, 'n_steps': 2048}

            model.myTrain(total_timesteps=timesteps, policy_used='CnnPolicy', arguments=policy_args)
            return

        if self.mode == 'test':
            model.myTest(model_num=self.model, episodes=self.episodes)
            return

    def deadly_corridor(self):
        print("\nDescription:\n"
              "The map is a corridor with shooting monsters on both sides (6 monsters in total). A green vest is "
              "placed at the opposite end of the corridor. The reward is proportional (negative or positive) to the "
              "change in the distance between the player and the vest. If the player ignores monsters on the sides "
              "and runs straight for the vest, he will be killed somewhere along the way.\n")

        model = Doom_Models(level=self.level, algorithm=self.algorithm, adjustments=True, use_curriculum=True)

        if self.mode == 'train':
            policy_args = None
            policy = None
            timesteps = 250000

            if self.algorithm == 'PPO':
                policy = 'CnnPolicy'
                policy_args = {'learning_rate': 0.00001,
                               'n_steps': 4096,
                               'ent_coef': 0.001}

            elif self.algorithm == 'DQN':
                policy = 'CnnPolicy'
                policy_args = {'learning_rate': 0.0001,
                               'buffer_size': 10_000,
                               'batch_size': 32,
                               'exploration_fraction': 0.5}

            model.myTrain(total_timesteps=timesteps, policy_used=policy, arguments=policy_args)
            return

        if self.mode == 'test':
            model.myTest(model_num=self.model, episodes=self.episodes)
            return

    def deathmatch(self):
        print("\nDescription:\n"
              "In this scenario, the agent is spawned in the random place of the arena filled with "
              "resources. A random monster is spawned every few seconds that will try to kill the player. The reward "
              "for killing a monster depends on its difficulty. The aim of the agent is to kill as many monsters as "
              "possible before the time runs out or it’s killed by monsters.\n")

        model = Doom_Models(level=self.level, algorithm=self.algorithm)

        if self.mode == 'train':
            policy_args = None
            policy = None
            timesteps = 200000

            if self.algorithm == 'PPO':
                policy = 'CnnPolicy'
                policy_args = {'learning_rate': 0.00001, 'n_steps': 4096}
            elif self.algorithm == 'DQN':
                policy = 'CnnPolicy'
                policy_args = {'learning_rate': 0.0001, 'buffer_size': 10_000, 'batch_size': 32}

            model.myTrain(total_timesteps=timesteps, policy_used=policy,
                          arguments=policy_args)
            return

        if self.mode == 'test':
            model.myTest(model_num=self.model, episodes=self.episodes)
            return

    def defend_the_line(self):
        print("\nDescription:\n"
              "The purpose of this scenario is to teach an agent that killing the monsters is GOOD and when monsters"
              "kill you is BAD. In addition, wasting ammunition is not very good either."
              "The agent is rewarded only for killing monsters, so it has to figure out the rest for itself."

              "The map is a rectangle. A player is spawned along the longer wall in the center. 3 melee-only and 3 "
              "shooting monsters are spawned along the opposite wall. Monsters are killed after a single shot, "
              "at first. After dying, each monster is respawned after some time and can endure more damage. The "
              "episode ends when the player dies (it’s inevitable because of limited ammo).\n")

        model = Doom_Models(level=self.level, algorithm=self.algorithm, adjustments=False, use_curriculum=False)

        if self.mode == 'train':
            policy_args = None
            policy = None
            timesteps = 250000

            if self.algorithm == 'PPO':
                policy = 'CnnPolicy'
                policy_args = {'learning_rate': 0.00001,
                               'n_steps': 4096,
                               'ent_coef': 0.001}

            elif self.algorithm == 'DQN':
                policy = 'CnnPolicy'
                policy_args = {'learning_rate': 0.0001,
                               'buffer_size': 10_000,
                               'batch_size': 32,
                               'exploration_fraction': 0.5}

            model.myTrain(total_timesteps=timesteps, policy_used=policy, arguments=policy_args)
            return

        if self.mode == 'test':
            model.myTest(model_num=self.model, episodes=self.episodes)
            return

    def health_gathering(self):
        pass

    def health_gathering_supreme(self):
        pass

    def my_way_home(self):
        pass

    def predict_position(self):
        pass

    def take_cover(self):
        pass

    def e1m1(self):
        pass
