from Classes.ViZDoom_Gym import ViZDoom_Gym
from Classes.TrainAndLog_Callback import TrainAndLog_Callback


def doom_Basic():
    CHKPNT_DIR = './train/train_basic'
    LOG_DIR = './logs/log_basic'


    doom = ViZDoom_Gym("basic", render=True)
    doom.close()


if __name__ == '__main__':
    doom_Basic()
