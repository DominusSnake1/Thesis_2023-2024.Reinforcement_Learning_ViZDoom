import signal
import sys


def mode_selector():
    args = sys.argv[1:]

    # Training Mode
    if len(args) == 2 and args[0] == '-mode' and args[1] == 'train':
        return 'train', -1
    # Testing Mode
    elif len(args) == 4 and args[0] == '-mode' and args[1] == 'test' and args[2] == '-model':
        if args[3] == 'YOUR_MODEL_HERE':
            raise Exception('Please specify the model name! (Pick from `Data/Train`)')
        else:
            return 'test', args[3]
