import torch
from Utils.Utils import mode_selector
from Models.Doom_PPO import Doom_PPO


def doom_Basic():
    choice, test_model = mode_selector()

    model = Doom_PPO(level='basic')

    if choice == 'train':
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.myTrain(device, verbose=1, learning_rate=0.0001, n_steps=2048)
    elif choice == 'test':
        model.myTest(test_model)


def doom_DefendTheCenter():
    choice, test_model = mode_selector()
    model = Doom_PPO(level='defend_the_center')

    if choice == 'train':
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.myTrain(device, verbose=1, learning_rate=0.0001, n_steps=2048, total_timesteps=100000)
    elif choice == 'test':
        model.myTest(test_model)


if __name__ == '__main__':
    # doom_Basic()
    doom_DefendTheCenter()
