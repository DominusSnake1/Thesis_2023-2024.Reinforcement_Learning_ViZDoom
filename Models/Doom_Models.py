from Classes.TrainAndLog_Callback import TrainAndLog_Callback
from Classes.ViZDoom_Gym import ViZDoom_Gym
from Other.Utils import timer
import torch
import time
from icecream import ic


class Doom_Models:
    def __init__(self, level: str, technique, render: bool = False):
        self.level = level
        self.technique = technique
        self.render = render

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.log_dir = f'./Data/Logs/log_{level}'

    @timer
    def myTrain(self, timesteps: int) -> None:
        print('Starting training...')
        print('(If you wish to stop training sooner, just press CTRL+C)\n')

        model = None
        callback = TrainAndLog_Callback(model_name=self.technique.algorithm,
                                        check_freq=25000,
                                        level=self.level,
                                        reward_shaping=self.technique.reward_shaping,
                                        curriculum=self.technique.curriculum_learning)

        log_name = f'{callback.get_formatted_datetime()}_{self.technique.algorithm}'

        doom = ViZDoom_Gym(level=self.level,
                           render=self.render,
                           reward_shaping=self.technique.reward_shaping,
                           curriculum=self.technique.curriculum_learning)

        algorithm = self.technique.algorithm[:3]

        if algorithm == 'PPO':
            from Training.CNNFeatureExtractor import CNNFeatureExtractor
            from stable_baselines3 import PPO

            model = PPO(env=doom,
                        policy=self.technique.policy,
                        learning_rate=self.technique.learning_rate,
                        n_steps=self.technique.n_steps,
                        ent_coef=self.technique.ent_coef,
                        batch_size=self.technique.batch_size,
                        gamma=self.technique.gamma,
                        clip_range=self.technique.clip_range,
                        gae_lambda=self.technique.gae_lambda,
                        policy_kwargs={
                            'features_extractor_class': CNNFeatureExtractor,
                            'features_extractor_kwargs': {
                                'number_of_actions': self.technique.number_of_actions
                            }
                        },
                        device=self.device,
                        tensorboard_log=self.log_dir,
                        verbose=1)

        if self.technique.curriculum_learning:
            from Training import CurriculumLearning

            last_model = CurriculumLearning.CurriculumLearning(timesteps=int(timesteps / 4),
                                                               render=self.render,
                                                               level=self.level,
                                                               log_name=log_name,
                                                               callback=callback,
                                                               model=model)

            model.load(last_model)

        model.learn(total_timesteps=timesteps,
                    callback=callback,
                    tb_log_name=log_name,
                    progress_bar=True,
                    reset_num_timesteps=True)

        doom.close()

    def myTest(self, model_name: str, episodes: int) -> None:
        doom = ViZDoom_Gym(self.level, render=True)
        model = None

        if self.technique.__class__.__name__ in ["PPO_Standard", "PPO_RewardShaping", "PPO_ResNet"]:
            from stable_baselines3 import PPO
            model = PPO.load(f'./Data/Train/train_{self.level}/{model_name}')

        avg_model_score = 0
        for episode in range(episodes):
            obs = doom.reset()
            done = False
            total_reward = 0
            while not done:
                if isinstance(obs, tuple):
                    obs = obs[0]
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
