import argparse


def level_selector(level: str) -> str:
    """
    Returns the ViZDoom level specified in the command line prompt using the custom argument '-lvl'.

    If no level or a wrong level is entered in the command line (with or without the '-lvl' tag),
    an exception is raised.

    :return: level
    """
    levels = ['basic',
              'defend_the_center',
              'deadly_corridor',
              'deathmatch',
              'defend_the_line']

    if level not in levels:
        exception_message = "Please select a level/scenario from the list:\n"

        for i, lvl in enumerate(levels, 1):
            exception_message += f"{i}. {lvl}\n"

        raise Exception(exception_message)

    return level


def mode_selector(mode: str) -> str:
    """
    Returns the mode specified in the command line prompt using the custom argument '-m'.

    If no mode or a wrong mode is entered in the command line (with or without the '-m' tag),
    an exception is raised.

    :return: mode
    """
    if mode not in ('train', 'test'):
        raise Exception("Please select a mode from 'train' or 'test'!")

    return mode


def technique_selector(technique: str) -> str:
    techniques = ['PPO_Standard',
                  'PPO_RewardShaping',
                  'PPO_Curriculum',
                  'PPO_RewardShaping_and_Curriculum',
                  'PPO_CustomCNN',
                  'DQN_Standard']

    if technique not in techniques:
        exception_message = "Please select a technique from the list:\n"

        for i, tech in enumerate(techniques, 1):
            exception_message += f"{i}. {tech}\n"

        raise Exception(exception_message)

    return technique


def modelAndEpisodes_selector(model: str, episodes: int) -> tuple:
    """
    Returns the model and the number of episodes specified in the command line prompt
    using the custom arguments '-mdl' and '-eps'.

    If no model is entered in the command line (with or without the '-mdl' tag), an exception is raised.

    If no number of episodes is entered in the command line (with or without the '-eps' tag), an exception is raised.

    :return: model and number of episodes
    """
    if int(episodes) <= 0:
        raise Exception("Please provide a positive number of episodes.")

    return model, int(episodes)


def parse_arguments():
    parser = argparse.ArgumentParser(description='ViZDoom Experiment Configuration')

    parser.add_argument('-lvl', '--level', type=str, required=True, help='Select the level/scenario')
    parser.add_argument('-m', '--mode', type=str, required=True, help='Choose the mode: train or test')
    parser.add_argument('-t', '--technique', type=str, required=True, help='Select the technique')
    parser.add_argument('-mdl', '--model', type=str, help='Specify the model name (required for testing)')
    parser.add_argument('-eps', '--episodes', type=int, help='Number of episodes to test (required for testing)')
    parser.add_argument('-r', '--render', action='store_true', help='Render the game')
    parser.add_argument('-d', '--display_rewards', action='store_true', help='Display rewards during training')

    args = parser.parse_args()

    config = {
        "level": level_selector(args.level),
        "mode": mode_selector(args.mode),
        "technique": technique_selector(args.technique),
        "model": None,
        "episodes": None,
        "render": args.render,
        "display_rewards": args.display_rewards
    }

    if config["mode"] == 'test':
        if not args.model or not args.episodes:
            parser.error("Testing requires both --model and --episodes arguments.")
        config["model"], config["episodes"] = modelAndEpisodes_selector(args.model, args.episodes)

    return config
