from Classes.TrainAndLog_Callback import TrainAndLog_Callback
from Classes.ViZDoom_Gym import ViZDoom_Gym
from Other.Utils import timer
import numpy as np
import torch
import time


class Doom_Models:
    def __init__(self, level, adjustments=False):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.level = level
        self.adjustments = adjustments
        self.log_dir = f'./Data/Logs/log_{level}'

    @timer
    def myTrain(self, algorithm, policy_used, arguments, total_timesteps):
        callback = TrainAndLog_Callback(model_name=algorithm, check_freq=10000, level=self.level,
                                        adjustments=self.adjustments)

        print('Starting training...')
        print('(If you wish to stop training sooner, just press CTRL+C)\n')

        doom = ViZDoom_Gym(self.level, adjustments=self.adjustments)
        model = None

        if algorithm == 'PPO':
            from stable_baselines3 import PPO
            model = PPO(policy=policy_used, env=doom, tensorboard_log=self.log_dir,
                        device=self.device, **arguments, verbose=1)
        elif algorithm == 'DQN':
            from stable_baselines3 import DQN
            model = DQN(policy=policy_used, env=doom, tensorboard_log=self.log_dir,
                        device=self.device, **arguments, verbose=1)

        model.learn(total_timesteps=total_timesteps, callback=callback)

        doom.close()

    def myTest(self, algorithm, model_num, episodes):
        doom = ViZDoom_Gym(self.level, render=True)
        model = None

        if algorithm == 'PPO':
            from stable_baselines3 import PPO
            model = PPO.load(f'./Data/Train/train_{self.level}/{model_num}')
        elif algorithm == 'DQN':
            from stable_baselines3 import DQN
            model = DQN.load(f'./Data/Train/train_{self.level}/{model_num}')

        avg_model_score = 0
        for episode in range(episodes):
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
            print(f"Episode \'{episode + 1}\' reward: {total_reward} pts")
            avg_model_score += total_reward
            time.sleep(1)
        avg_model_score /= episodes
        print(f"Average reward for these {episodes} episodes is: {avg_model_score} pts")

        doom.close()
