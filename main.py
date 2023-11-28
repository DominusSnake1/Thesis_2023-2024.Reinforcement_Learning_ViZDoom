import torch
from Classes.ViZDoom_Gym import ViZDoom_Gym
from Classes.TrainAndLog_Callback import TrainAndLog_Callback
from stable_baselines3 import PPO


def doom_Basic():
    print(torch.__version__)
    print(torch.cuda.device_count())
    CHKPNT_DIR = './train/train_basic'
    LOG_DIR = './logs/log_basic'

    callback = TrainAndLog_Callback(check_freq=10000, save_path=CHKPNT_DIR)

    doom = ViZDoom_Gym("basic", render=False)

    model = PPO('CnnPolicy', env=doom, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=256)

    doom.close()


if __name__ == '__main__':
    doom_Basic()
