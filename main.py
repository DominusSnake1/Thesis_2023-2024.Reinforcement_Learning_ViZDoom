import time
import numpy as np
import torch
from Utils.Utils import mode_selector
from Classes.ViZDoom_Gym import ViZDoom_Gym
from Classes.TrainAndLog_Callback import TrainAndLog_Callback
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy


def doom_Basic():
    def train(LOG_DIR, device, logger):
        doom = ViZDoom_Gym("basic")

        model = PPO('CnnPolicy', env=doom, tensorboard_log=LOG_DIR, verbose=1, learning_rate=0.0001, n_steps=2048,
                    device=device)
        model.learn(total_timesteps=100000, callback=logger)

        doom.close()

    def test(model_num):
        doom = ViZDoom_Gym("basic", True)

        model = PPO.load(f'./Data/Train/train_basic/{model_num}')

        # mean_reward, _ = evaluate_policy(model, doom, n_eval_episodes=100)
        # print('mean_reward: ', mean_reward)

        avg_model_score = 0
        for episode in range(5):
            obs = doom.reset()
            done = False
            total_reward = 0
            while not done:
                if isinstance(obs, tuple):
                    obs = obs[0]
                obs = np.concatenate([obs] * 3, axis=0)
                action, _ = model.predict(obs)
                obs, reward, done, _, info = doom.step(action)
                time.sleep(0.05)
                total_reward += reward
            print(f"Total reward for episode {episode} is {total_reward}")
            avg_model_score += total_reward
            time.sleep(1)
        avg_model_score /= 5
        print(f"Average reward for these 5 episodes is: {avg_model_score}")

        doom.close()

    choice, test_model = mode_selector()

    if choice == 'Train':
        device = use_Cuda_Or_Cpu()
        callback = TrainAndLog_Callback(check_freq=10000, save_path='./Data/Train/train_basic')

        train('./Data/Logs/log_basic', device, callback)
    elif choice == 'test':
        test(test_model)


def use_Cuda_Or_Cpu():
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


if __name__ == '__main__':
    doom_Basic()
