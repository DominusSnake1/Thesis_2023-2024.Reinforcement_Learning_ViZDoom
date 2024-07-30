import sys


def level_selector() -> str:
    """
    Returns the ViZDoom level specified in the command line prompt using the custom argument '-lvl'.

    If no level or a wrong level is entered in the command line (with or without the '-lvl' tag),
    an exception is raised.

    :return: level
    """
    args = sys.argv[1:]
    levels = ['basic', 'defend_the_center', 'deadly_corridor', 'deathmatch', 'defend_the_line']

    if args[0] != '-lvl':
        raise Exception('In order to choose a level you must use \'-lvl\' tag after ./main.py.')

    if (args[1] == 'SELECT_LEVEL') or (args[1] not in levels):
        exception_message = "Please select a level/scenario from the list:\n"

        for i, level in enumerate(levels, 1):
            exception_message += f"{i}. {level}\n"

        raise Exception(exception_message)

    return args[1]


def mode_selector() -> str:
    """
    Returns the mode specified in the command line prompt using the custom argument '-m'.

    If no mode or a wrong mode is entered in the command line (with or without the '-m' tag),
    an exception is raised.

    :return: mode
    """
    args = sys.argv[1:]

    if args[2] != '-m':
        raise Exception("In order to choose a mode you must use \'-m\' after selecting a level.")

    if args[3] not in ('train', 'test'):
        raise Exception("Please select a mode from 'train' or 'test'!")

    return args[3]


def technique_selector() -> str:
    args = sys.argv[1:]
    techniques = ['PPO_Standard', 'PPO_RewardShaping', 'PPO_CurriculumLearning', 'DQN_Standard']

    if args[4] != '-t':
        raise Exception("In order to choose a technique you must use \'-t\' after selecting a mode.")

    if (args[5] == 'YOUR_TECHNIQUE') or (args[5] not in techniques):
        exception_message = "Please select a technique from the list:\n"

        for i, technique in enumerate(techniques, 1):
            exception_message += f"{i}. {technique}\n"

        raise Exception(exception_message)

    return args[5]


def modelAndEpisodes_selector() -> tuple:
    """
    Returns the model and the number of episodes specified in the command line prompt
    using the custom arguments '-mdl' and '-eps'.

    If no model is entered in the command line (with or without the '-mdl' tag), an exception is raised.

    If no number of episodes is entered in the command line (with or without the '-eps' tag), an exception is raised.

    :return: model and number of episodes
    """
    args = sys.argv[1:]

    if (len(args) < 7) or (args[6] != '-mdl'):
        raise Exception("In order to choose a model you must use \'-mdl\' after selecting a technique.")

    if (len(args) < 8) or (args[7] == 'YOUR_MODEL_HERE'):
        raise Exception("Please select a model from the \'Data/Train/YOURCHOSENLEVEL/YOURCHOSENMODEL\'.")

    if (len(args) < 9) or (args[8] != '-eps'):
        raise Exception("In order to choose how many episodes to test you must use \'-eps\' after selecting a model.")

    if (len(args) < 10) or ((args[9] == 'X') or (int(args[9]) <= 0)):
        raise Exception("Please provide a positive number of episodes.")

    return args[7], int(args[9])
