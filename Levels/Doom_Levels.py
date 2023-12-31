import sys
import torch
from Models.Doom_PPO import Doom_PPO


def level_selector():
    """
    Takes the Level as an argument and prints the list of available levels if the selected level is unavailable.

    :return: The selected level for training or testing.
    """
    args = sys.argv[1:]

    levels = ['basic', 'defend_the_center', 'deadly_corridor']

    if len(args) >= 2 and args[0] == '-level':
        if (args[1] == 'SELECT_LEVEL') or (args[1] not in levels):
            raise Exception('Please select a level/scenario from the list:\n'
                            '1. basic\n'
                            '2. defend_the_center\n'
                            '3. deadly_corridor\n')

        return args[1]


def mode_selector():
    """
    Takes the "mode", "model" and "episode number" as an argument.

    :return: Training/Testing Mode, Model and Number of Episodes
    """
    args = sys.argv[1:]

    # Training Mode
    if len(args) == 4 and args[2] == '-mode' and args[3] == 'train':
        return 'train', -1, -1

    # Testing Mode
    if len(args) == 8 and args[2] == '-mode' and args[3] == 'test' and args[4] == '-model' and args[6] == '-eps':
        if args[5] == 'YOUR_MODEL_HERE':
            raise Exception('Please specify the model name! (Pick from `Data/Train`)')

        if (args[7] == 'X') or (args[7] <= '0'):
            raise Exception('Please provide a positive number of episodes.')

        return 'test', args[5], int(args[7])

    raise Exception("Please pick a mode from 'train' or 'test'!")


class Doom_Levels:
    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.level = level_selector()
        self.mode, self.test_model, self.episodes = mode_selector()

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

        model = Doom_PPO(level='basic')

        if self.mode == 'train':
            model.myTrain(self.device, verbose=1, learning_rate=0.0001, n_steps=2048, total_timesteps=100000)
        elif self.mode == 'test':
            model.myTest(self.test_model, episodes=self.episodes)

    def defend_the_center(self):
        """
        Loads the `defend_the_center` level in either "Training" or "Testing" mode
        as well as give a simple description of the level.
        """
        print("\nDescription:\n"
              "The map is a large circle. A player is spawned in the exact center. 5 melee-only, monsters are spawned "
              "along the wall. Monsters are killed after a single shot. After dying, each monster is respawned after "
              "some time. The episode ends when the player dies (it’s inevitable because of limited ammo).\n")

        model = Doom_PPO(level='defend_the_center')

        if self.mode == 'train':
            model.myTrain(self.device, verbose=1, learning_rate=0.0001, n_steps=2048, total_timesteps=100000)
        elif self.mode == 'test':
            model.myTest(self.test_model, episodes=self.episodes)

    def deadly_corridor(self):
        print("\nDescription:\n"
              "The map is a corridor with shooting monsters on both sides (6 monsters in total). A green vest is "
              "placed at the opposite end of the corridor. The reward is proportional (negative or positive) to the "
              "change in the distance between the player and the vest. If the player ignores monsters on the sides "
              "and runs straight for the vest, he will be killed somewhere along the way.\n")

        model = Doom_PPO(level='deadly_corridor')

        if self.mode == 'train':
            model.myTrain(self.device, verbose=1, learning_rate=0.0001, n_steps=2048, total_timesteps=100000)
        elif self.mode == 'test':
            model.myTest(self.test_model, episodes=self.episodes)
