import torch
from Classes.ViZDoom_Gym import ViZDoom_Gym
from Classes.TrainAndLog_Callback import TrainAndLog_Callback
from stable_baselines3 import PPO


def doom_Basic():
    dev = select_device()

    CHKPNT_DIR = './train/train_basic'
    LOG_DIR = './logs/log_basic'

    callback = TrainAndLog_Callback(check_freq=10000, save_path=CHKPNT_DIR)

    doom = ViZDoom_Gym("basic")

    model = PPO('CnnPolicy', env=doom, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=256, device=dev)
    model.learn(total_timesteps=100000, callback=callback)

    doom.close()


def select_device():
    if torch.cuda.is_available():
        print(f"Cuda is AVAILABLE with {torch.cuda.device_count()} device(s) available.")
        print("Here is the list of available devices:")
        for i in range(torch.cuda.device_count()):
            print(f"{i+1}. {torch.cuda.get_device_name(i)}")

        ans = (input(f"\nDo you want to use CUDA/GPU instead of CPU? (y/n)\n").lower().strip() == 'y')

        if ans:
            return torch.device("cuda")

    return torch.device("cpu")


if __name__ == '__main__':
    doom_Basic()
