import sys


def mode_selector():
    args = sys.argv[1:]

    # Training Mode
    if len(args) == 2 and args[0] == '-mode' and args[1] == 'Train':
        return 'Train', -1
    # Testing Mode
    elif len(args) == 4 and args[0] == '-mode' and args[1] == 'test' and args[2] == '-model':
        return 'test', args[3]
