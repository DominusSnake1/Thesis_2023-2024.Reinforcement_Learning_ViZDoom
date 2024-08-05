from Classes.TrainAndLog_Callback import TrainAndLog_Callback
from Classes.ViZDoom_Gym import ViZDoom_Gym
from Other.Utils import timer
import torch
import time
from icecream import ic


class Doom_Models:
    def __init__(self, level: str, render: bool, display_rewards: bool = False):
        self.level = level
        self.render = render
        self.selected_technique = None
        self.display_rewards = display_rewards

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.log_dir = f'./Data/Logs/log_{level}'

    def set_technique(self, technique):
        self.selected_technique = technique

    @timer
    def myTrain(self, timesteps: int) -> None:
        print('Starting training...')
        print('(If you wish to stop training sooner, just press CTRL+C)\n')

        model = None
        callback = TrainAndLog_Callback(model_name=self.selected_technique.algorithm,
                                        check_freq=25000,
                                        level=self.level,
                                        reward_shaping=self.selected_technique.reward_shaping,
                                        curriculum=self.selected_technique.curriculum_learning)

        log_name = f'{callback.get_formatted_datetime()}_{self.selected_technique.algorithm}'

        doom = ViZDoom_Gym(level=self.level,
                           render=self.render,
                           display_rewards=self.display_rewards,
                           reward_shaping=self.selected_technique.reward_shaping,
                           curriculum=self.selected_technique.curriculum_learning)

        algorithm = self.selected_technique.algorithm[:3]

        if algorithm == 'PPO':
            from Training.CNNFeatureExtractor import CNNFeatureExtractor
            from stable_baselines3 import PPO

            model = PPO(env=doom,
                        policy=self.selected_technique.policy,
                        learning_rate=self.selected_technique.learning_rate,
                        n_steps=self.selected_technique.n_steps,
                        ent_coef=self.selected_technique.ent_coef,
                        batch_size=self.selected_technique.batch_size,
                        gamma=self.selected_technique.gamma,
                        clip_range=self.selected_technique.clip_range,
                        gae_lambda=self.selected_technique.gae_lambda,
                        policy_kwargs={
                            'features_extractor_class': CNNFeatureExtractor,
                            'features_extractor_kwargs': {
                                'number_of_actions': self.selected_technique.number_of_actions
                            }
                        },
                        device=self.device,
                        tensorboard_log=self.log_dir,
                        verbose=1)

        elif algorithm == 'DQN':
            from stable_baselines3 import DQN

            model = DQN(env=doom,
                        policy=self.selected_technique.policy,
                        learning_rate=self.selected_technique.learning_rate,
                        buffer_size=self.selected_technique.buffer_size,
                        learning_starts=self.selected_technique.learning_starts,
                        batch_size=self.selected_technique.batch_size,
                        tau=self.selected_technique.tau,
                        gamma=self.selected_technique.gamma,
                        gradient_steps=self.selected_technique.gradient_steps,
                        exploration_fraction=self.selected_technique.exploration_fraction,
                        exploration_initial_eps=self.selected_technique.exploration_initial_eps,
                        exploration_final_eps=self.selected_technique.exploration_final_eps,
                        max_grad_norm=self.selected_technique.max_grad_norm,
                        device=self.device,
                        tensorboard_log=self.log_dir,
                        verbose=1)

        if self.selected_technique.curriculum_learning:
            from Training import CurriculumLearning

            model = CurriculumLearning.CurriculumLearning(timesteps=int(timesteps / 4),
                                                          render=self.render,
                                                          level=self.level,
                                                          log_name=log_name,
                                                          callback=callback,
                                                          model=model)

        model.learn(total_timesteps=timesteps,
                    callback=callback,
                    tb_log_name=log_name,
                    progress_bar=True,
                    reset_num_timesteps=True)

        doom.close()

    def myTest(self, trained_model_name: str, episodes: int) -> None:
        doom = ViZDoom_Gym(self.level,
                           render=self.render,
                           display_rewards=self.display_rewards,
                           reward_shaping=self.selected_technique.reward_shaping)
        trained_model = None

        technique = self.selected_technique.__class__.__name__[:3]

        if technique == "PPO":
            from stable_baselines3 import PPO
            trained_model = PPO.load(f'./Data/Train/train_{self.level}/{trained_model_name}')

        elif technique == "DQN":
            from stable_baselines3 import DQN
            trained_model = DQN.load(f'./Data/Train/train_{self.level}/{trained_model_name}')

        avg_model_score = 0
        for episode in range(episodes):
            obs = doom.reset()
            done = False
            total_reward = 0
            while not done:
                if isinstance(obs, tuple):
                    obs = obs[0]
                action, _ = trained_model.predict(obs)
                obs, reward, done, _, info = doom.step(action)
                time.sleep(0.05)
                total_reward += reward
            print(f"Episode \'{episode + 1}\' reward: {total_reward} pts")
            avg_model_score += total_reward
            time.sleep(1)
        avg_model_score /= episodes
        print(f"Average reward for these {episodes} episodes is: {avg_model_score} pts")

        doom.close()
