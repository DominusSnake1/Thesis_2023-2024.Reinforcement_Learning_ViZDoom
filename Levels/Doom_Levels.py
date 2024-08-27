from Models.Doom_Models import Doom_Models
from Training import Techniques
from Other import CMD


class Doom_Levels:
    def __init__(self):
        configuration = CMD.parse_arguments()

        self.level = configuration['level']
        self.mode = configuration['mode']

        self.technique = configuration['technique']

        if self.mode == 'test':
            self.trained_model, self.episodes = configuration['model'], configuration['episodes']

        self.model = Doom_Models(level=self.level,
                                 render=configuration['render'],
                                 display_rewards=configuration['display_rewards'])

    def basic(self):
        print("\nDescription:\n"
              "The map is a rectangle with gray walls, ceiling, and floor. The player is spawned along the longer "
              "wall in the center. A red, circular monster is spawned randomly somewhere along the opposite wall. A "
              "player can only (config) go left/right and shoot. 1 hit is enough to kill the monster. The episode "
              "finishes when the monster is killed or on timeout.\n")

        selected_technique = None

        if self.technique == 'PPO_Standard':
            selected_technique = Techniques.PPO_Standard()
        elif self.technique == 'PPO_RewardShaping':
            selected_technique = Techniques.PPO_RewardShaping()

        self.model.set_technique(technique=selected_technique)

        if self.mode == 'train':
            # model.myTrain(timesteps=100000)
            self.model.myTrain(timesteps=250000)
        elif self.mode == 'test':
            self.model.myTest(trained_model_name=self.trained_model, episodes=self.episodes)

    def defend_the_center(self, number_of_actions: int = 3):
        print("\nDescription:\n"
              "The map is a large circle. A player is spawned in the exact center. 5 melee-only, monsters are spawned "
              "along the wall. Monsters are killed after a single shot. After dying, each monster is respawned after "
              "some time. The episode ends when the player dies (it’s inevitable because of limited ammo).\n")

        selected_technique = None

        if self.technique == 'PPO_Standard':
            selected_technique = Techniques.PPO_Standard(number_of_actions=number_of_actions)
        elif self.technique == 'PPO_RewardShaping':
            selected_technique = Techniques.PPO_RewardShaping(number_of_actions=number_of_actions,
                                                              batch_size=256,
                                                              n_steps=4096,
                                                              learning_rate=0.0002,
                                                              ent_coef=0.0001,
                                                              clip_range=0.1,
                                                              gamma=0.995,
                                                              gae_lambda=0.95)

        self.model.set_technique(technique=selected_technique)

        if self.mode == 'train':
            # model.myTrain(timesteps=100000)
            self.model.myTrain(timesteps=250000)
        elif self.mode == 'test':
            self.model.myTest(trained_model_name=self.trained_model, episodes=self.episodes)

    def deadly_corridor(self, actions: int = 7, doom_skill: int = 5):
        print("\nDescription:\n"
              "The map is a corridor with shooting monsters on both sides (6 monsters in total). A green vest is "
              "placed at the opposite end of the corridor. The reward is proportional (negative or positive) to the "
              "change in the distance between the player and the vest. If the player ignores monsters on the sides "
              "and runs straight for the vest, he will be killed somewhere along the way.\n")

        selected_technique = None

        if self.technique == 'PPO_Standard':
            selected_technique = Techniques.PPO_Standard(number_of_actions=actions,
                                                         batch_size=256,
                                                         n_steps=8192,
                                                         learning_rate=0.0001,
                                                         ent_coef=0.01,
                                                         clip_range=0.1,
                                                         gamma=0.95,
                                                         gae_lambda=0.9)
        elif self.technique == 'PPO_RewardShaping':
            selected_technique = Techniques.PPO_RewardShaping(number_of_actions=actions,
                                                              batch_size=256,
                                                              n_steps=4096,
                                                              learning_rate=0.0002,
                                                              ent_coef=0.001,
                                                              clip_range=0.05,
                                                              gamma=0.99,
                                                              gae_lambda=0.9)

        elif self.technique == 'PPO_Curriculum':
            selected_technique = Techniques.PPO_Curriculum(number_of_actions=actions,
                                                           default_skill=doom_skill,
                                                           batch_size=256,
                                                           n_steps=8192,
                                                           learning_rate=0.0002,
                                                           ent_coef=0.01,
                                                           clip_range=0.1,
                                                           gamma=0.95,
                                                           gae_lambda=0.9)

        elif self.technique == 'PPO_RewardShaping_and_Curriculum':
            selected_technique = Techniques.PPO_RewardShaping_and_Curriculum(number_of_actions=actions,
                                                                             default_skill=doom_skill,
                                                                             batch_size=256,
                                                                             n_steps=8192,
                                                                             learning_rate=0.0002,
                                                                             ent_coef=0.01,
                                                                             clip_range=0.1,
                                                                             gamma=0.95,
                                                                             gae_lambda=0.9)
        elif self.technique == 'PPO_CustomCNN':
            selected_technique = Techniques.PPO_CustomCNN(number_of_actions=actions)

        elif self.technique == 'DQN_Standard':
            selected_technique = Techniques.DQN_Standard(number_of_actions=actions,
                                                         buffer_size=10_000)

        self.model.set_technique(technique=selected_technique)

        if self.mode == 'train':
            self.model.myTrain(timesteps=5000)
        elif self.mode == 'test':
            self.model.myTest(trained_model_name=self.trained_model, episodes=self.episodes)

    def deathmatch(self, number_of_actions: int = 17, doom_skill: int = 3):
        print("\nDescription:\n"
              "In this scenario, the agent is spawned in the random place of the arena filled with "
              "resources. A random monster is spawned every few seconds that will try to kill the player. The reward "
              "for killing a monster depends on its difficulty. The aim of the agent is to kill as many monsters as "
              "possible before the time runs out or it’s killed by monsters.\n")

        selected_technique = None

        if self.technique == 'PPO_Standard':
            selected_technique = Techniques.PPO_Standard(batch_size=256,
                                                         n_steps=8192,
                                                         learning_rate=0.0003,
                                                         ent_coef=0.001,
                                                         clip_range=0.1,
                                                         gamma=0.95,
                                                         gae_lambda=0.9)
        elif self.technique == 'PPO_RewardShaping':
            selected_technique = Techniques.PPO_RewardShaping()
        elif self.technique == 'PPO_Curriculum':
            selected_technique = Techniques.PPO_Curriculum(default_skill=doom_skill)
        elif self.technique == 'PPO_CustomCNN':
            selected_technique = Techniques.PPO_CustomCNN(number_of_actions=number_of_actions)

        self.model.set_technique(technique=selected_technique)

        if self.mode == 'train':
            self.model.myTrain(timesteps=400000)
        elif self.mode == 'test':
            self.model.myTest(trained_model_name=self.trained_model, episodes=self.episodes)

    def defend_the_line(self):
        print("\nDescription:\n"
              "The purpose of this scenario is to teach an agent that killing the monsters is GOOD and when monsters"
              "kill you is BAD. In addition, wasting ammunition is not very good either. The agent is rewarded only"
              "for killing monsters, so it has to figure out the rest for itself."

              "The map is a rectangle. A player is spawned along the longer wall in the center. 3 melee-only and 3 "
              "shooting monsters are spawned along the opposite wall. Monsters are killed after a single shot, "
              "at first. After dying, each monster is respawned after some time and can endure more damage. The "
              "episode ends when the player dies (it’s inevitable because of limited ammo).\n")

        selected_technique = None

        if self.technique == 'PPO_Standard':
            selected_technique = Techniques.PPO_Standard()
        elif self.technique == 'PPO_RewardShaping':
            selected_technique = Techniques.PPO_RewardShaping()

        self.model.set_technique(technique=selected_technique)

        if self.mode == 'train':
            self.model.myTrain(timesteps=250000)
        elif self.mode == 'test':
            self.model.myTest(trained_model_name=self.trained_model, episodes=self.episodes)

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
