import sys


def level_selector():
    args = sys.argv[1:]
    levels = ['basic', 'defend_the_center', 'deadly_corridor']

    if args[0] != '-lvl':
        raise Exception('In order to choose a level you must use \'-lvl\' after ./main.py.')

    if (args[1] == 'SELECT_LEVEL') or (args[1] not in levels):
        raise Exception('Please select a level/scenario from the list:\n'
                        '1. basic\n'
                        '2. defend_the_center\n'
                        '3. deadly_corridor\n')

    return args[1]


def mode_selector():
    args = sys.argv[1:]

    if args[2] != '-m':
        raise Exception("In order to choose a mode you must use \'-m\' after selecting a level.")

    if args[3] not in ('train', 'test'):
        raise Exception("Please select a mode from 'train' or 'test'!")

    return args[3]


def algorithm_selector():
    args = sys.argv[1:]
    algorithms = ['PPO', 'DQN']

    if args[4] != '-alg':
        raise Exception("In order to choose an algorithm you must use \'-alg\' after selecting a mode.")

    if (args[5] == 'YOUR_ALGORITHM') or (args[5] not in algorithms):
        raise Exception("Please select an algorithm from the list:\n"
                        "1. PPO\n"
                        "2. DQN\n")

    return args[5]


def modelAndEpisodes_selector():
    args = sys.argv[1:]

    if args[6] != '-mdl':
        raise Exception("In order to choose a model you must use \'-mdl\' after selecting an algorithm.")

    if args[7] == 'YOUR_MODEL_HERE':
        raise Exception("Please select a model from the \'Data/Train/YOURCHOSENLEVEL/YOURCHOSENMODEL\'.")

    if args[8] != '-eps':
        raise Exception("In order to choose how many episodes to test you must use \'-eps\' after selecting a model.")

    if (args[9] == 'X') or (int(args[9]) <= 0):
        raise Exception("Please provide a positive number of episodes.")

    return args[7], int(args[9])
