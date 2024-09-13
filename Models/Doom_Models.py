from Classes.TrainAndLog_Callback import TrainAndLog_Callback
from Classes.ViZDoom_Gym import ViZDoom_Gym
import Training.CurriculumLearning as CL
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

        # Initialize the callback in order to save the progress of the model.
        callback = TrainAndLog_Callback(model_name=self.selected_technique.algorithm,
                                        check_freq=50000,
                                        level=self.level,
                                        reward_shaping=self.selected_technique.reward_shaping,
                                        curriculum=self.selected_technique.curriculum_learning)

        # Initialize the name of the log file according to the datetime and the technique used.
        log_name = f'{callback.get_formatted_datetime()}_{self.selected_technique.algorithm}'

        model = self.init_PPO_model()

        if self.selected_technique.curriculum_learning:
            CL.CurriculumLearning(timesteps=timesteps,
                                  render=self.render,
                                  display_rewards=self.display_rewards,
                                  reward_shaping=self.selected_technique.reward_shaping,
                                  level=self.level,
                                  default_skill=self.selected_technique.doom_skill,
                                  log_name=log_name,
                                  callback=callback,
                                  model=model)

        else:
            # Initialize the environment.
            doom = ViZDoom_Gym(level=self.level,
                               difficulty=self.selected_technique.doom_skill,
                               render=self.render,
                               display_rewards=self.display_rewards,
                               reward_shaping=self.selected_technique.reward_shaping)

            model.set_env(env=doom)

            model.learn(total_timesteps=timesteps,
                        callback=callback,
                        tb_log_name=log_name,
                        progress_bar=True,
                        reset_num_timesteps=True)

            doom.close()

    def myTest(self, trained_model_name: str, episodes: int) -> None:
        # Initialize the environment.
        doom = ViZDoom_Gym(level=self.level,
                           difficulty=self.selected_technique.doom_skill,
                           render=self.render,
                           display_rewards=self.display_rewards,
                           reward_shaping=self.selected_technique.reward_shaping)

        # Load the trained model from the files.
        trained_model = self.load_trained_model(trained_model_name)

        # Prints the model's hyperparameters.
        self.print_model_parameters(model=trained_model, model_name=trained_model_name)

        # Test the model and print the average score.
        average_model_score = self.test_model(episodes=episodes, model_for_testing=trained_model, environment=doom)
        print(f"Average reward for these {episodes} episodes is: {average_model_score} pts")

        # Close the environment.
        doom.close()

    def init_PPO_model(self):
        from stable_baselines3 import PPO
        from stable_baselines3.common import policies

        return PPO(env=CL.Blank_Env(self.selected_technique.number_of_actions),
                   policy=policies.ActorCriticCnnPolicy,
                   learning_rate=self.selected_technique.learning_rate,
                   n_steps=self.selected_technique.n_steps,
                   ent_coef=self.selected_technique.ent_coef,
                   batch_size=self.selected_technique.batch_size,
                   gamma=self.selected_technique.gamma,
                   clip_range=self.selected_technique.clip_range,
                   gae_lambda=self.selected_technique.gae_lambda,
                   policy_kwargs={'features_extractor_kwargs':
                                      {'features_dim': self.selected_technique.number_of_actions}
                                  },
                   device=self.device,
                   tensorboard_log=self.log_dir,
                   verbose=1)

    def load_trained_model(self, trained_model_name):
        """
        Loads the trained model from the saved models in ./Data/Train according to the level.
        This trained model will be used in testing in order to validate the model.

        If the model specified by the user doesn't exist, an exception is raised.

        :param trained_model_name: The model specified by the user

        :return: The trained model
        """
        algorithm = self.selected_technique.__class__.__name__[:3]

        if algorithm == "PPO":
            from stable_baselines3 import PPO
            return PPO.load(f'./Data/Train/train_{self.level}/{trained_model_name}')

        elif algorithm == "DQN":
            from stable_baselines3 import DQN
            return DQN.load(f'./Data/Train/train_{self.level}/{trained_model_name}')

        else:
            raise Exception("The model specified by the user does not exist. Please choose a different model.")

    def print_model_parameters(self, model, model_name):
        print("=" * 10, model_name, "=" * 10)
        print("learning_rate:", model.learning_rate)
        print("n_steps:", model.n_steps)
        print("ent_coef:", model.ent_coef)
        print("batch_size:", model.batch_size)
        print("clip_range:", model.clip_range(1))
        print("gamma:", model.gamma)
        print("gae_lamda:", model.gae_lambda)
        print("=" * 40)

    def test_model(self, episodes: int, model_for_testing, environment: ViZDoom_Gym):
        """
        Tests the already trained for a specified number of episodes,
        prints the reward for each episode and returns the average reward for those episodes.

        :param episodes: The number of episodes to test
        :param model_for_testing: The trained model for testing
        :param environment: The environment in which the model is being tested

        :return: The average reward for all episodes
        """
        avg_model_score = 0
        for episode in range(episodes):
            obs = environment.reset()
            done = False
            total_reward = 0
            while not done:
                if isinstance(obs, tuple):
                    obs = obs[0]
                action, _ = model_for_testing.predict(obs)
                obs, reward, done, _, info = environment.step(action)
                time.sleep(0.05)
                total_reward += reward
            print(f"Episode \'{episode + 1}\' reward: {round(total_reward, 3)} pts")
            avg_model_score += total_reward
            time.sleep(1)
        return round((avg_model_score / episodes), 3)
