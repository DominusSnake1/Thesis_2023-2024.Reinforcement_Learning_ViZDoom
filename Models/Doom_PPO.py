import time
import numpy as np
from stable_baselines3 import PPO
from Classes.TrainAndLog_Callback import TrainAndLog_Callback
from Classes.ViZDoom_Gym import ViZDoom_Gym


class Doom_PPO:
    def __init__(self, level):
        self.level = level
        self.log_dir = f'./Data/Logs/log_{level}'
        self.callback = TrainAndLog_Callback(check_freq=10000, save_path=f'./Data/Train/train_{level}')

    def myTrain(self, device, verbose, learning_rate, n_steps, total_timesteps):
        print('Starting training...')
        print('(If you wish to stop training sooner, just press CTRL+C)\n')

        doom = ViZDoom_Gym(self.level)

        model = PPO('CnnPolicy', env=doom, tensorboard_log=self.log_dir, verbose=verbose,
                    learning_rate=learning_rate, n_steps=n_steps, device=device)

        model.learn(total_timesteps=total_timesteps, callback=self.callback)

        doom.close()

    def myTest(self, model_num, episodes):
        doom = ViZDoom_Gym(self.level, True)

        model = PPO.load(f'./Data/Train/train_{self.level}/{model_num}')

        # mean_reward, _ = evaluate_policy(model, doom, n_eval_episodes=100)
        # print('mean_reward: ', mean_reward)

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
            print(f"Episode \'{episode+1}\' reward: {total_reward} pts")
            avg_model_score += total_reward
            time.sleep(1)
        avg_model_score /= episodes
        print(f"Average reward for these {episodes} episodes is: {avg_model_score} pts")

        doom.close()
